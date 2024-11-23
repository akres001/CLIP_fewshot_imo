import torch
from tqdm import tqdm
import torch.nn.functional as F
from clip import clip
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(clip_model.visual.conv1.weight.device)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1)#.cuda()
    return clip_weights


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def pre_load_features(cfg, split, adapt_model, loader, seed, model_type='', device=''):

    if cfg['load_pre_feat'] == False:
        features, features_original, labels, features_proj = [], [], [], []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader)):
                
                images = batch['img']
                target = batch['label']
                    
                images, target = images.to(device), target.to(device)
                image_features = adapt_model.encode_image(images, apply_adapter=True)
                # original features for zero-shot CLIP do NOT require adapted features
                image_features_original = adapt_model.encode_image(images, apply_adapter=False)
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features_original /= image_features_original.norm(dim=-1, keepdim=True)
                
                features.append(image_features.cpu())
                features_original.append(image_features_original.cpu())
                
                labels.append(target.cpu())

        features, features_original, labels = torch.cat(features) , torch.cat(features_original), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + f"seed_{seed}_" + model_type + split + "_f.pt")
        torch.save(features_original, cfg['cache_dir'] + "/" + f"seed_{seed}_" + model_type + split + "_f_original.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + f"seed_{seed}_" + model_type + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + f"seed_{seed}_" + model_type + split + "_f.pt", map_location="cpu")
        labels = torch.load(cfg['cache_dir'] + "/" + f"seed_{seed}_" + model_type + split + "_l.pt", map_location="cpu")
        features_original = torch.load(cfg['cache_dir'] + "/" + f"seed_{seed}_" + model_type + split + "_f_original.pt", map_location="cpu")
    
    return features, features_original, labels

def build_cache_model(cfg, adapt_model, train_loader_cache, seed,  model_type='', device=''):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_keys_original = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                train_features_original = []
                train_features_proj = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, batch in enumerate(tqdm(train_loader_cache)):
                    images = batch['img']
                    target = batch['label']
                    
                    images = images.to(device)
                    # image_features, image_features_projected = mymodel.encode_image(images)
                    image_features = adapt_model.encode_image(images, apply_adapter=True)
                    image_features_original = adapt_model.encode_image(images, apply_adapter=False)
                    
                    train_features.append(image_features)
                    train_features_original.append(image_features_original)
                        
                    if augment_idx == 0:
                        target = target#.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
                cache_keys_original.append(torch.cat(train_features_original, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        
        cache_keys_original = torch.cat(cache_keys_original, dim=0).mean(dim=0)
        cache_keys_original /= cache_keys_original.norm(dim=-1, keepdim=True)
        cache_keys_original = cache_keys_original.permute(1, 0)
                
        cache_values = F.one_hot(torch.cat(cache_values, dim=0))#.half()

        torch.save(cache_keys, cfg['cache_dir'] + '/' + f"augmt_{cfg['augment_epoch']}_seed_{seed}_" + model_type + 'keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_keys_original, cfg['cache_dir'] + '/'  + f"augmt_{cfg['augment_epoch']}_seed_{seed}_" + model_type + 'keys_original_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/'  + f"augmt_{cfg['augment_epoch']}_seed_{seed}_" + model_type +  'values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/'  + f"augmt_{cfg['augment_epoch']}_seed_{seed}_" + model_type + 'keys_' + str(cfg['shots']) + "shots.pt", map_location="cpu")
        cache_keys_original = torch.load(cfg['cache_dir'] + '/'  + f"augmt_{cfg['augment_epoch']}_seed_{seed}_" + model_type + 'keys_original_' + str(cfg['shots']) + "shots.pt",  map_location="cpu")
        cache_values = torch.load(cfg['cache_dir'] + '/'  + f"augmt_{cfg['augment_epoch']}_seed_{seed}_" + model_type +  'values_' + str(cfg['shots']) + "shots.pt",  map_location="cpu")
    
    return cache_keys, cache_keys_original, cache_values



def scale_(x, target):
    
    y = (x - x.min()) / (x.max() - x.min())
    y *= target.max() - target.min()
    y += target.min()
    
    return y

def cls_acc(output, target, topk=1):
    pred = np.argmax(output, axis=1)
    
    # Check if predictions match the target
    correct = pred == target.reshape(1, -1)
    
    # Calculate accuracy
    acc = correct[:topk].reshape(-1).sum(0)
    acc = 100 * acc / target.shape[0]
    
    return acc

def get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution, bs = 100):
    
    kl_divs_sim = torch.zeros((test_image_class_distribution.shape[0], train_image_class_distribution.shape[0])).to(train_image_class_distribution.device)

    for i in tqdm(range(test_image_class_distribution.shape[0]//bs), desc='kl divergence'):
        curr_batch = test_image_class_distribution[i*bs : (i+1)*bs]
        repeated_batch = torch.repeat_interleave(curr_batch, train_image_class_distribution.shape[0], dim=0)    
        q = train_image_class_distribution
        q_repeated = torch.cat([q]*bs)
        kl = repeated_batch * (repeated_batch.log() - q_repeated.log())
        kl = kl.sum(dim=-1)
        kl = kl.view(bs, -1)
        kl_divs_sim[ i*bs : (i+1)*bs , : ] = kl  
        
    kl_divs_sim = kl_divs_sim.to(train_image_class_distribution.device)

    return kl_divs_sim


def get_kl_div_sims(test_features, val_features, train_features, clip_weights, temp=0.5):

    train_image_class_distribution = train_features.T @ clip_weights
    train_image_class_distribution = nn.Softmax(dim=-1)(train_image_class_distribution/temp)

    test_image_class_distribution = test_features @ clip_weights
    test_image_class_distribution = nn.Softmax(dim=-1)(test_image_class_distribution/temp)

    val_image_class_distribution = val_features @ clip_weights
    val_image_class_distribution = nn.Softmax(dim=-1)(val_image_class_distribution/temp)

    train_kl_divs_sim = None
    test_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution)
    val_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, val_image_class_distribution)

    return train_kl_divs_sim, test_kl_divs_sim, val_kl_divs_sim

def pred_tipadapter(cliplogits, feats, c_keys, c_values, labels, alpha, beta):
    
    affinity = feats.type(torch.float) @ c_keys.type(torch.float)
    logits_cache = ((-1) * (beta - beta * affinity)).exp() @ c_values.type(torch.float)

    tipa_logits = cliplogits + logits_cache * alpha
    acc = cls_acc(tipa_logits.detach().cpu().numpy(), labels.detach().cpu().numpy())
    return acc


def pred_tip_x(cliplogits, kllogits, newknow, c_values, labels, alpha, beta, gamma):
        
    logits_cache = ((-1) * (beta - beta * newknow)).exp() @ c_values.type(torch.float)
    logits_tipx = cliplogits + kllogits * gamma + logits_cache * alpha
    acc = cls_acc(logits_tipx.detach().cpu().numpy(), labels.detach().cpu().numpy())

    return acc



def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            # when using `DistributedSampler`, i.e., when sample is NOT None, 
            # we set up shuffle = False as it's set already in `DistributedSampler`
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders 



class Dataset_openimages(Dataset):
    
    def __init__(self, data, transform, root, ltoi, dict_out=False): 
        self.root = root
        self.data = data.values
        self.transform = transform
        self.dict_out = dict_out
        self.ltoi = ltoi
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):    
        
        image = Image.open(self.root  + self.data[index][0] + ".jpg").convert("RGB")
        label = self.ltoi[self.data[index][1]]
        image = self.transform(image)
        
        if self.dict_out:
            return {'img': image, 'label' : label}
        return (image, label)