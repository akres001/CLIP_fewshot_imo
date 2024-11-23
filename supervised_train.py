import os
from tqdm import tqdm
import json
import copy
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
import re
from PIL import Image
import pandas as pd

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

import sys

from adapter_utils import CLIP_Adapter
import utils

import argparse
from yacs.config import CfgNode as CN
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from clip import clip
from clip import model as clip_model_package
import torch

configs = {}
    
if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/app/datasets/open_imagesv4/images_classes/", help="path to dataset")
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--images_path", type=str, default="/app/datasets/open_imagesv4/oi_data.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ffn_adapter_scalar", type=float, default=0.03)
        
    args = parser.parse_args()

    print("Arguments : ", args)

    df_labels = pd.read_csv(args.images_path)
    
    torch.manual_seed(args.seed)
    
    EPOCHS = args.epochs
    
    SAVE_TO_FOLDER = f"models_checkpoint/OI_size200_lr_{args.lr}_epochs{args.epochs}_{args.attempt}/"
    os.makedirs(SAVE_TO_FOLDER, exist_ok=True)    
        
    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    print("Loading model..")
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    print("Building model..")        
    adapter_config = {'ffn_adapt' : True, 'ffn_adapter_init_option' : 'lora', 'ffn_adapter_scalar' : args.ffn_adapter_scalar,
                      'ffn_num' : 64, 'ffn_adapter_layernorm_option' : 'none', 'd_model' : 768}
        
    adapter_config = CN(adapter_config)
    kwargs = {'adapter_config' : adapter_config}
    model = clip.build_model(state_dict or model.state_dict(), **kwargs).float().cuda()
        
    BICUBIC = InterpolationMode.BICUBIC

    pretrain_transform = Compose([                        
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),  
            ToTensor(),
            # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])   
    
    ltoi = {o:i for i, o in enumerate(df_labels.DisplayName.unique())}
    itol = {i : o for i, o in enumerate(df_labels.DisplayName.unique())}
    
    open_images_data = df_labels[['ImageID', 'DisplayName', 'type']]

    openimages_ds = utils.Dataset_openimages(open_images_data, pretrain_transform, root=args.root, ltoi=ltoi)   
    openimages_loader = utils.create_loader([openimages_ds],[None], batch_size=[args.batch_size], num_workers=[0], is_trains=[True], collate_fns=[None])[0]
    
    ce_loss = nn.CrossEntropyLoss()
    
    learner = CLIP_Adapter(model, **{'out_class' :  len(ltoi), 'backbone' : 'ViT'})
    learner = learner.float()
    learner.cuda()
    learner.train()
    
    # freeze image encoders, so we only update the adapter part
    for name, param in learner.image_encoder.named_parameters():
        if 'adaptmlp' not in name:
            param.requires_grad_(False)


    print()
    print("parameters that will update:\n")
    for name, param in learner.named_parameters():
        if param.requires_grad == True:
            print(name)
            
    lr = args.lr    
    opt = torch.optim.AdamW(learner.parameters(), lr=lr)
    
    configs['lr'] = lr
    configs['adapter_config'] = adapter_config
    configs['epochs'] = EPOCHS
    
    with open(SAVE_TO_FOLDER + "/configs.txt", "w") as f:
        f.write(str(configs))

    all_losses = []
    for epoch in range(EPOCHS):
        losses = []
        for ii, batch in enumerate(tqdm(openimages_loader)):
            images = batch[0].cuda()
            labels = batch[1].cuda()
            labels = F.one_hot(labels,  len(ltoi)).float()

            predictions = learner(images)
            loss = ce_loss(predictions, labels)
            opt.zero_grad()
            loss.backward()
            losses.append(loss.item())
            opt.step()
            
            if ii % 100 == 0:
                print(opt.param_groups[0]["lr"])
        
        print(f"Average loss epoch {epoch} : {np.mean(losses)}")
        all_losses.append(losses)
        torch.save(learner.state_dict(), SAVE_TO_FOLDER + f'model_epoch_{epoch}.pt')
        
        with open(SAVE_TO_FOLDER + "/losses.txt", "a") as f:
            f.write(f"epoch {epoch} : {np.mean(losses)}")