import os
from tqdm import tqdm
import json
import copy
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import torch
import pandas as pd
import random
import copy
import sys
import argparse
import matplotlib.pyplot as plt
import pickle
from PIL import Image

# from dassl.config import get_cfg_default
from yacs.config import CfgNode as CN
from sklearn.metrics.pairwise import cosine_similarity

from dinov1 import utils as dinoutils
from dinov1 import vision_transformer as vits

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset

from clip import clip
import utils

from torch.distributed import init_process_group
import torch.distributed as dist

import math
from pathlib import Path

torch.set_num_threads(15)



def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = dinoutils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    # for it, (batch) in enumerate(metric_logger.log_every(data_loader, 10, header)):
    for it, (batch) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        
        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in batch['img']]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2], apply_adapter=True)  # only the 2 global views pass through the teacher
            student_output = student(images, apply_adapter=True)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = dinoutils.clip_gradients(student, args.clip_grad)
            dinoutils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = dinoutils.clip_gradients(student, args.clip_grad)
            dinoutils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/app/datasets", help="path to dataset")
    parser.add_argument("--oi_root", type=str, default="/app/datasets/open_imagesv4/images_classes/", help="path to dataset")
    parser.add_argument("--images_path", type=str, default="/app/datasets/open_imagesv4/oi_data.csv")
    parser.add_argument('--use_dino_transformer', action='store_true', help="use ViT from dino instead of CLIP - NOTE : this is just to test functionality")
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--only_val', action='store_true')
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--restore_checkpoint', action='store_true')
    parser.add_argument('--save_best_last', action='store_true')
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    # https://github.com/facebookresearch/dino/issues/104#issuecomment-903829270
    # incrase momentum to 0.9995 with small batch size  
    parser.add_argument("--momentum_teacher", type=float, default=0.9995, help="output directory")
    parser.add_argument('--adapter_train', action='store_true')
    parser.add_argument("--optimizer", type=str, default="adamw", help="output directory")
    parser.add_argument("--epochs", type=int, default=100, help="output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="output directory")
    parser.add_argument("--output_dir", type=str, default="/app/fewshot/Saved_Adapter_Experiments/DINO_imagenet/", help="output directory")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument("--clip_grad", type=float, default=0.3, help="only positive value enables a fixed seed")
    parser.add_argument("--freeze_last_layer", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument("--root_vit", type=str, default=os.path.expanduser("~/.cache/clip"), help='directory for clip vit model. In Jade use /jmain02/home/J2AD018/mrt08/axk45-mrt08/external_models') 
    
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=50, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    
    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""") 
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=2e-06, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    
    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.25, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=10, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.25),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    
    args = parser.parse_args()
    args = CN(vars(args))
    
    
    if args.distributed:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
    if args.distributed:
        device = int(os.environ["LOCAL_RANK"])
    else:
        device = 'cuda'
    
    if args.distributed:
        seed = args.seed + dinoutils.get_rank()
    else:
        seed = args.seed
        
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, args.root_vit)

    print("Loading model..")
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    print("Building model..")
    if args.adapter_train:
        adapter_config = {'ffn_adapt' : True, 'ffn_adapter_init_option' : 'lora', 'ffn_adapter_scalar' : 0.1, 
                                  'ffn_num' : 64, 'ffn_adapter_layernorm_option' : 'none', 'd_model' : 768}
        adapter_config = CN(adapter_config)
    else:
        adapter_config = None
    kwargs = {'adapter_config' : adapter_config, 'dino' : True}
    model = clip.build_model(state_dict or model.state_dict(), **kwargs).float()#.cuda()
    
    
    EPOCHS = args.epochs
    lr = args.lr #0.0005 # * args.batch_size/256
    
    configs = {}
    configs['lr'] = lr
    configs['adapter_config'] = adapter_config
    configs['epochs'] = EPOCHS
    
    print("Configs", configs)
    
    # print("Config path", os.path.join(args.output_dir, "configs.txt"))
    with open(os.path.join(args.output_dir, "configs.txt"), "w") as f:
        f.write(str(configs))

    
    kwargs = {'backbone' : 'ViT', 
              'warmup_epochs' : args.warmup_epochs,"weight_decay": args.weight_decay, "weight_decay_end": args.weight_decay_end,
               'out_dim' : args.out_dim, 'local_crops_number' : args.local_crops_number, 'warmup_teacher_temp' : args.warmup_teacher_temp, 'teacher_temp' : args.teacher_temp, "min_lr": args.min_lr,
               'warmup_teacher_temp_epochs' : args.warmup_teacher_temp_epochs, 'epochs' : EPOCHS, 'use_bn_in_head' : False, 'embed_dim' : 512, 'norm_last_layer' : True}


    transform = dinoutils.DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        kwargs['local_crops_number'],
    )
    

            
    df_labels = pd.read_csv(args.images_path)

    ltoi = {o:i for i, o in enumerate(df_labels.DisplayName.unique())}
    itol = {i : o for i, o in enumerate(df_labels.DisplayName.unique())}
    len(ltoi)

    open_images_data = df_labels[['ImageID', 'DisplayName', 'type']]
    train_dataset = utils.Dataset_openimages(open_images_data, transform, root=args.oi_root, ltoi=ltoi, dict_out=True)   
    
        
    
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)

        data_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    sampler=sampler,
                    batch_size=args.batch_size,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True,
                )  
    else:
        data_loader = build_data_loader(
                        cfg,
                        sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                        data_source=train_dataset,
                        batch_size=args.batch_size,
                        tfm=transform,
                        is_train=True,
                        dataset_wrapper=None
                    )
    
    if args.use_dino_transformer:
        assert not args.adapter_train
        # multi-crop wrapper handles forward with inputs of different resolutions
        kwargs['embed_dim'] = 768
        student = vits.__dict__['vit_base'](
            patch_size=16,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__['vit_base'](patch_size=16)
        
        student = dinoutils.MultiCropWrapper(student, dinoutils.DINOHead(
                kwargs['embed_dim'],
                kwargs['out_dim'],
                use_bn=kwargs['use_bn_in_head'],
                norm_last_layer=kwargs['norm_last_layer'],
            ))

        teacher = dinoutils.MultiCropWrapper(
            teacher,
            dinoutils.DINOHead(kwargs['embed_dim'], kwargs['out_dim'], kwargs['use_bn_in_head']),
        )
        
    else:
        # multi-crop wrapper handles forward with inputs of different resolutions
        student = dinoutils.MultiCropWrapper(copy.deepcopy(model.visual), dinoutils.DINOHead(
                kwargs['embed_dim'],
                kwargs['out_dim'],
                use_bn=kwargs['use_bn_in_head'],
                norm_last_layer=kwargs['norm_last_layer'],
            ))

        teacher = dinoutils.MultiCropWrapper(
            copy.deepcopy(model.visual),
            dinoutils.DINOHead(kwargs['embed_dim'], kwargs['out_dim'], kwargs['use_bn_in_head']),
        )
        

    dino_loss = dinoutils.DINOLoss(
            kwargs['out_dim'],
            kwargs['local_crops_number'] + 2,  # total number of crops = 2 global crops + local_crops_number
            kwargs['warmup_teacher_temp'],
            kwargs['teacher_temp'],
            kwargs['warmup_teacher_temp_epochs'],
            kwargs['epochs'],
            distributed=args.distributed
    ).to(device)

    # move networks to gpu
    student, teacher = student.to(device), teacher.to(device)
    
    find_unused_parameters = False
    if args.adapter_train: 
        find_unused_parameters = True
        
    if (args.distributed) & (dinoutils.has_batchnorms(student)):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[device], find_unused_parameters=find_unused_parameters)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    
    if args.distributed:
        student = nn.parallel.DistributedDataParallel(student, device_ids=[device], find_unused_parameters=find_unused_parameters)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
        
    for p in teacher.parameters():
        p.requires_grad = False
    
    # freeze image encoders, so we only update the adapter part
    if args.adapter_train:
        for name, param in student.named_parameters():
            if 'adaptmlp' not in name:
                param.requires_grad_(False)

    
    # ============ preparing optimizer ... ============
    params_groups = dinoutils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = dinoutils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    
    print("WORLD SIZE : ", dinoutils.get_world_size())
    # ============ init schedulers ... ============
    lr_schedule = dinoutils.cosine_scheduler(
        lr * (args.batch_size * dinoutils.get_world_size()) / 256.,  # linear scaling rule
        kwargs['min_lr'],
        args.epochs, len(data_loader),
        warmup_epochs=kwargs['warmup_epochs'],
    )
    wd_schedule = dinoutils.cosine_scheduler(
        kwargs['weight_decay'],
        kwargs['weight_decay_end'],
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = dinoutils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")
    
    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.restore_checkpoint:
        dinoutils.restart_from_checkpoint(
            os.path.join(args.output_dir, 'checkpoint.pth'),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
    start_epoch = to_restore["epoch"]
    
    with (Path(args.output_dir) / "args.txt").open("w") as f:
        f.write(json.dumps(args) + "\n")

    with (Path(args.output_dir) / "kwargs.txt").open("w") as f:
        f.write(json.dumps(kwargs) + "\n")
    
    best_area = 1.
    for epoch in range(start_epoch, EPOCHS):
        data_loader.sampler.set_epoch(epoch)
        if args.distributed:
            rank = dist.get_rank() == 0
        else:
            rank = True
            
        if rank and epoch == 0:
            print("Student params update: ")
            for name, param in student.named_parameters():
                if param.requires_grad == True:
                    print(name)

            print("Teacher params update: ")
            for name, param in teacher.named_parameters():
                if param.requires_grad == True:
                    print(name)
        
        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)
        
        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        dinoutils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if rank: # main process
            dinoutils.save_on_master(save_dict, os.path.join(args.output_dir, f'model_{epoch}.pth'))

        if args.distributed:
            dist.barrier()