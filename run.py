import numpy as np
import argparse
import os
import torch
import random
from yacs.config import CfgNode as CN

# from dassl.data import DataManager
from dassl.data.datasets.build import build_dataset
from dassl.data.transforms.transforms import build_transform
import datasets.ucf101
import datasets.sun397
import datasets.oxford_flowers
import datasets.oxford_pets
import datasets.food101
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.imagenet
import datasets.stanford_cars
import datasets.caltech101
import datasets.eurosat
import datasets.plantdoc
import datasets.cub
import datasets.stanford_dogs

from dassl.data.data_manager import build_data_loader
from dassl.config import get_cfg_default
import pickle
import utils

from templates import IMAGENET_TEMPLATES_SELECT, CUSTOM_TEMPLATES
from clip import clip
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms

torch.set_num_threads(20)

SKIP = []
LOAD_DATA = False

N_search_alpha = 200
N_search_beta = 20
N_search_gamma = 50
alpha_max = 20
beta_max = 20
gamma_max = 30
    
DEVICE = f'cuda:0'
SAVE_TO_FOLDER = "search/"
os.makedirs(SAVE_TO_FOLDER, exist_ok=True)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetname", type=str)
    parser.add_argument("--root", type=str, default="/app/datasets/")
    parser.add_argument("--num_shots", type=str, default=-1)
    parser.add_argument("--num_augment", type=int, default=10)
    parser.add_argument("--model_checkpoint", type=str)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    print(args)
    args.config_file = "configs/trainers/Adapter/adam_lr2e-4_B256_ep200_ViT16.yaml"

    cfg = get_cfg_default()

    cfg.merge_from_file(args.config_file)
    # cfg.TRAINER.NAME = 'TaskRes'
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.SEED = args.seed
    cfg.DATASET.ROOT = args.root
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 4
    cfg.DATALOADER.TEST.BATCH_SIZE = 128
    
    ds = args.datasetname
    
    cache_name = SAVE_TO_FOLDER + "/cache/"
        
    logs = f"{ds} \n\n"
    
    if not LOAD_DATA:
        
        print("-------------> KEYS FROM PATH", args.model_path)
        from collections import OrderedDict
        keys_changed = torch.load(args.model_path)
        new_model = OrderedDict()
        if 'teacher' in keys_changed:
            print("UNSUPERVISED LOADING TEACHER")
            
            for k, p in keys_changed['teacher'].items():
                new_k = k.replace("backbone.", "")
                if 'predictor' in k or 'head.' in k: continue
                new_model[new_k] = p
            
        else:
            for k, p in keys_changed.items():
                new_k = k.replace("image_encoder.", "")
                if 'predictor' in k: continue
                new_model[new_k] = p
    
        
        backbone_name = "ViT-B/16"
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
        
        from yacs.config import CfgNode
        config_file = '/'.join(args.model_path.split("/")[:-1]) + "/configs.txt"
        
        with open(config_file, "r") as f:
            config_file_loaded = f.read()
        
        adapter_config = eval(config_file_loaded)['adapter_config']
        print("adapter_config", adapter_config)
        
        kwargs = {'adapter_config' : adapter_config}

        print("Loading original model..")
        try:
            # loading JIT archive
            clip_adapted = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        
            
        clip_adapted = clip.build_model(state_dict or clip_adapted.state_dict(), **kwargs).float()
        
        print("Loading new weights: ", clip_adapted.visual.load_state_dict(new_model))
        
        clip_adapted.to(DEVICE)
    else:
        clip_adapted = None
    
    
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATASET.NUM_SHOTS = 1
    cfg.DATASET.NAME = ds # 
    cfg.SEED = int(args.seed)
        
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    
    dataset = build_dataset(cfg)

    print("Getting textual features as CLIP's classifier.")
    if cfg.DATASET.NAME == 'ImageNet':
        clip_weights = utils.clip_classifier(dataset.classnames, IMAGENET_TEMPLATES_SELECT, clip_adapted)
    else:
        clip_weights = utils.clip_classifier(dataset.classnames, [CUSTOM_TEMPLATES[cfg.DATASET.NAME]], clip_adapted)

    clip_weights = clip_weights.to(DEVICE)
   
    
    results = {}
    
    if args.num_shots != -1:
        all_shots = [int(args.num_shots)]
    else:
        all_shots = [1,2,4,8,16]
    
    print(f"Searching in {all_shots} shots")
    
    for n_shots in all_shots:
        
        ds_shot = ds + "_" + str(n_shots)
        results[ds_shot] = {}
        print(f"****** Testing {ds_shot} dataset ******")
        cfg.DATASET.NUM_SHOTS = n_shots
        cfg.DATASET.NAME = ds 
        dataset = build_dataset(cfg)

        tfm_test = build_transform(cfg, is_train=False)
        
        train_tranform = transforms.Compose([
                    transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                ])
        
        logs += f"Number of shots {n_shots} \n" 

        train_loader_x = build_data_loader(
                    cfg,
                    sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                    data_source=dataset.train_x,
                    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                    n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                    n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
                    tfm=train_tranform,
                    is_train=False,
                    dataset_wrapper=None
                )

        val_loader = build_data_loader(
                        cfg,
                        sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                        data_source=dataset.val,
                        batch_size=32,
                        tfm=tfm_test,
                        is_train=False,
                        dataset_wrapper=None
                    )

        test_loader = build_data_loader(
                        cfg,
                        sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                        data_source=dataset.test,
                        batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                        tfm=tfm_test,
                        is_train=False,
                        dataset_wrapper=None
                    )
        
        
        print("\nConstructing cache model by few-shot visual features and labels.")

        cfg.load_cache = LOAD_DATA
        cfg.augment_epoch = args.num_augment
        cfg.cache_dir = cache_name
        cfg.shots = n_shots 

       
        cache_keys, cache_keys_original, cache_values = utils.build_cache_model(cfg, clip_adapted, train_loader_x, cfg.SEED, model_type=ds, device=DEVICE)
        if args.num_shots == -1:
            if n_shots == 1:
                cfg.load_pre_feat = LOAD_DATA
                # only compute test features once per dataset
                test_features, test_features_original, test_labels = utils.pre_load_features(cfg, "test", clip_adapted, test_loader, cfg.SEED, model_type=ds, device=DEVICE)
                val_features, val_features_original, val_labels = utils.pre_load_features(cfg, "val", clip_adapted, val_loader, cfg.SEED, model_type=ds, device=DEVICE)
        else:
            cfg.load_pre_feat = LOAD_DATA
            # only compute test features once per dataset
            test_features, test_features_original, test_labels = utils.pre_load_features(cfg, "test", clip_adapted, test_loader, cfg.SEED, model_type=ds, device=DEVICE)
            val_features, val_features_original, val_labels = utils.pre_load_features(cfg, "val", clip_adapted, val_loader, cfg.SEED, model_type=ds, device=DEVICE)
        
        
        cache_keys_original = cache_keys_original.to(DEVICE).type(torch.float)
        cache_keys = cache_keys.to(DEVICE).type(torch.float)
        cache_values = cache_values.to(DEVICE).type(torch.float)
        
        test_features = test_features.to(DEVICE).type(torch.float)
        test_features_original = test_features_original.to(DEVICE).type(torch.float)
        test_labels = test_labels.to(DEVICE).type(torch.float)
        
        val_features = val_features.to(DEVICE).type(torch.float)
        val_features_original = val_features_original.to(DEVICE).type(torch.float)
        val_labels = val_labels.to(DEVICE).type(torch.float)
        
        
        if not ('tipx' in SKIP and 'tipxmg' in SKIP):
            train_kl_divs_sim, test_kl_divs_sim, val_kl_divs_sim = utils.get_kl_div_sims(test_features_original, 
                                                                                   val_features_original, 
                                                                                   cache_keys_original, 
                                                                                   clip_weights.float())
            

        if not ('tipx' in SKIP and 'tipxmg' in SKIP):
            test_kl_divs_sim = test_kl_divs_sim.to(DEVICE)
            val_kl_divs_sim = val_kl_divs_sim.to(DEVICE)



        # Zero-shot CLIP
        clip_logits_val = 100. * val_features_original @ clip_weights
        clip_logits_test = 100. * test_features_original @ clip_weights
        acc = utils.cls_acc(clip_logits_test.detach().cpu().numpy(), test_labels.detach().cpu().numpy())
        
        print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))
        results[ds_shot]['ZS'] = acc
        logs += "Zero-Shot accuracy: {:.4f}. ".format(acc)
        # Tip-Adapter
        if 'tip' not in SKIP:
            best_acc = 0
            for alpha in tqdm(np.linspace(0, alpha_max, N_search_alpha), desc='Tip-Adapter'):
                for beta in np.linspace(0, beta_max, N_search_beta):

                    acc = utils.pred_tipadapter(clip_logits_val, val_features_original, cache_keys_original, 
                                          cache_values, val_labels, alpha, beta)

                   

                    if acc > best_acc:
                        best_acc = acc
                        best_alpha = alpha
                        best_beta = beta

            print("Best on validation: ")
            print(best_acc, best_alpha, best_beta)

            acc = utils.pred_tipadapter(clip_logits_test, test_features_original, cache_keys_original, 
                                  cache_values, test_labels, best_alpha, best_beta)


            print("**** Tip-Adapter's ORIGINAL test accuracy: {:.2f}. ****\n".format(acc))
            results[ds_shot]['ORIGINAL'] = (acc, (best_alpha, best_beta))

            logs += "Tip-Adapter's ORIGINAL test accuracy: {:.4f}. ".format(acc)
            logs += f"best alpha and beta {best_alpha, best_beta}\n" 

        # Tip-Adapter corrected
        if 'tipmg' not in SKIP:
            best_acc = 0
            for alpha in tqdm(np.linspace(0, alpha_max, N_search_alpha), desc='Tip-Adapter MG'):
                for beta in np.linspace(0, beta_max, N_search_beta):

                    acc = utils.pred_tipadapter(clip_logits_val, val_features, cache_keys, 
                                          cache_values, val_labels, alpha, beta)
                   

                    if acc > best_acc:
                        best_acc = acc
                        best_alpha = alpha
                        best_beta = beta

            print("Best on validation: ")
            print(best_acc, best_alpha, best_beta)

            acc = utils.pred_tipadapter(clip_logits_test, test_features, cache_keys, 
                                  cache_values, test_labels, best_alpha, best_beta)

            print("**** Tip-Adapter's ADAPTED test accuracy: {:.2f}. ****\n".format(acc))
            results[ds_shot]['ADAPTED'] = (acc, (best_alpha, best_beta))    

            logs += "Tip-Adapter's ADAPTED test accuracy: {:.4f}. ".format(acc)
            logs += f"best alpha and beta {best_alpha, best_beta}\n" 

        
        # Tip-X
        if 'tipx' not in SKIP:
            new_knowledge = val_features_original @ cache_keys_original

            neg_affs = utils.scale_(val_kl_divs_sim, new_knowledge)
            kl_logits_val = (-neg_affs) @ cache_values

            best_acc = 0
            for best_alpha_tipx in tqdm(np.linspace(0, alpha_max, N_search_alpha), desc='Tip-X'):
                for best_beta_tipx in np.linspace(0, beta_max, N_search_beta):
                    for best_gamma_tipx in np.linspace(0, gamma_max, N_search_gamma):

                        acc = utils.pred_tip_x(clip_logits_val, kl_logits_val, new_knowledge, 
                                                cache_values, val_labels, 
                                                best_alpha_tipx, best_beta_tipx, best_gamma_tipx)

                        if acc > best_acc:
                            best_acc = acc
                            best_alpha = best_alpha_tipx
                            best_beta = best_beta_tipx
                            best_gamma = best_gamma_tipx

            print("Best on validation: ")
            print(best_acc, best_alpha, best_beta, best_gamma)

            new_knowledge = test_features_original @ cache_keys_original

            neg_affs = utils.scale_(test_kl_divs_sim, new_knowledge)
            kl_logits_test = (-neg_affs) @ cache_values

            acc = utils.pred_tip_x(clip_logits_test, kl_logits_test, new_knowledge, 
                             cache_values, test_labels, 
                             best_alpha, best_beta, best_gamma)


            print("**** Tip-X test accuracy: {:.2f}. ****\n".format(acc))
            results[ds_shot]['tip_x'] = (acc, (best_alpha, best_beta, best_gamma))    


            logs += "Tip-X test accuracy: {:.4f}. ".format(acc)
            logs += f"best alpha and beta and gamma {best_alpha, best_beta, best_gamma}\n" 
        
   
        # Tip-X corrected
        if 'tipxmg' not in SKIP:
            new_knowledge_adapted = val_features @ cache_keys
            new_knowledge = val_features_original @ cache_keys_original

            neg_affs = utils.scale_(val_kl_divs_sim, new_knowledge_adapted)
            kl_logits_val = (-neg_affs) @ cache_values

            best_acc = 0
            for best_alpha_tipx in tqdm(np.linspace(0, alpha_max, N_search_alpha), desc='Tip-X MG'):
                for best_beta_tipx in np.linspace(0, beta_max, N_search_beta):
                    for best_gamma_tipx in np.linspace(0, gamma_max, N_search_gamma):

                        acc = utils.pred_tip_x(clip_logits_val, kl_logits_val, new_knowledge_adapted, 
                                                cache_values, val_labels, 
                                                best_alpha_tipx, best_beta_tipx, best_gamma_tipx)
                        if acc > best_acc:
                            best_acc = acc
                            best_alpha = best_alpha_tipx
                            best_beta = best_beta_tipx
                            best_gamma = best_gamma_tipx

            print("Best on validation: ")
            print(best_acc, best_alpha, best_beta, best_gamma)

            new_knowledge_adapted = test_features @ cache_keys
            new_knowledge = test_features_original @ cache_keys_original

            neg_affs = utils.scale_(test_kl_divs_sim, new_knowledge_adapted)
            kl_logits_test = ( -neg_affs) @ cache_values

            acc = utils.pred_tip_x(clip_logits_test, kl_logits_test, new_knowledge_adapted, 
                             cache_values, test_labels, 
                             best_alpha, best_beta, best_gamma)


            print("**** Tip-X Adapted test accuracy: {:.2f}. ****\n".format(acc))
            results[ds_shot]['tip_x_adapted'] = (acc, (best_alpha, best_beta, best_gamma))    
            print(results[ds_shot])

            logs += "Tip-X Adapted test accuracy: {:.4f}. ".format(acc)
            logs += f"best alpha and beta and gamma {best_alpha, best_beta, best_gamma}\n" 

    
        os.makedirs(args.save_dir, exist_ok=True)

        # append to logs
        with open(args.save_dir + f"logs_{ds}.txt", "a") as f:
            f.write(logs)
            
        logs = f"\n\n"
    
    if args.num_shots == -1:
        # if all shots save results
        with open(args.save_dir + f"results_{ds}.pkl", "wb") as f:
            pickle.dump(results, f)