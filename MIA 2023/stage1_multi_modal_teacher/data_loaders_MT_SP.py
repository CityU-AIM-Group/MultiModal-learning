"""
Author: Xing Xiaohan
Date: 2021.12.31
Construct dataset for the mean teacher framework.
Transform the pathology image twice as input for the student and mean teacher model, respectively.
Currently, use the same omic input for the student and mean teacher.
Memory banks are also included for the CRD loss.
"""
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets, transforms

from utils import mixed_collate

## meilu
import getpass
import torchvision.transforms.functional as TF
import random
#https://github.com/Algy/fast-slic
from fast_slic import Slic # pip install fast_slic 

def omic_transform(omic_data, transform='drop', rate=0.2):
    # print("original_data:", omic_data) ## [bs, dim]
    mask = np.random.binomial(1, rate, omic_data.shape)

    if transform == "drop":
        omic_data = omic_data * (1.0 - mask)
 
    ### the vime augmentation is from the paper: 
    ### VIME: Extending the success of self- And semi-supervised learning to tabular domain
    elif transform == "vime":
        no, dim = omic_data.shape  
        # Randomly (and column-wise) shuffle data
        x_bar = np.zeros([no, dim])
        for i in range(dim):
            idx = np.random.permutation(no)
            x_bar[:, i] = omic_data[idx, i]            
        # Corrupt samples
        omic_data = omic_data * (1-mask) + x_bar * mask
    
    return torch.tensor(omic_data).type(torch.FloatTensor)



def pathomic_dataloader(opt, data):

    custom_data_loader = Pathomic_InstanceSample(opt, data, split='train', mode=opt.mode)
    print("number of training samples:", len(custom_data_loader))
    train_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader, batch_size=opt.batch_size, 
        num_workers=4, shuffle=True, collate_fn=mixed_collate, drop_last=True)
    
    n_data = len(custom_data_loader)

    test_data_loader = PathomicDatasetLoader(opt, data, split='test', mode=opt.mode)
    print("number of testing samples:", len(test_data_loader))
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data_loader, batch_size=opt.batch_size, 
        num_workers=4, shuffle=False, collate_fn=mixed_collate)

    return train_loader, test_loader, n_data



def pathomic_patches_dataloader(opt, data):
    """
    Load the test set, each ROI image corresponds to 9 patch inputs with the size of 512*512.
    """
    test_data_loader = PathomicDatasetLoader(opt, data, split='test', mode=opt.mode)
    print("number of testing patches:", len(test_data_loader))
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data_loader, batch_size=opt.batch_size, 
        num_workers=4, shuffle=False, collate_fn=mixed_collate)

    return test_loader


#################################
# Dataloader without memory bank
#################################
class PathomicDatasetLoader(Dataset):
    def __init__(self, opt, data, split, mode='omic'):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        # print(data[split]['x_path'])
        self.X_path = data[split]['x_path']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.g = data[split]['g']
        self.mode = mode
        self.dataroot = opt.dataroot
        for i in range(len(self.X_path)):
            self.X_path[i] = self.X_path[i].replace("./data/TCGA_GBMLGG/", "/")

        self.superpixels = {}
        # generate superpixel
        if self.mode == "path" or self.mode == 'pathpath' or self.mode == 'pathomic':
            #meilu
            for index in range(len(self.X_path)):
                if getpass.getuser() == 'meiluzhu2':
                    image_path = '/home/meiluzhu2/data/MIM'+ self.X_path[index].split('data')[-1]
                    single_X_path = Image.open(image_path).convert('RGB')
                    slic = Slic(num_components=opt.num_superpixels, compactness=10)
                    sp_mask = slic.iterate(np.array(single_X_path)) #superpixel mask 
                    sp_mask = Image.fromarray(sp_mask)                    
                    self.superpixels[self.X_path[index]] = sp_mask
                else:
                    single_X_path = Image.open(self.dataroot + self.X_path[index]).convert('RGB') 
                    slic = Slic(num_components=opt.num_superpixels, compactness=10)
                    sp_mask = slic.iterate(np.array(single_X_path)) #superpixel mask 
                    sp_mask = Image.fromarray(sp_mask)                    
                    self.superpixels[self.X_path[index]] = sp_mask  
            print('The number of superpixel:',len(self.superpixels))
        
        ## meilu
        self.opt = opt
        
        if opt.label_dim == 2:
            ### 改成二分类，将标签中的1改为0, 标签中的2改为1
            label = self.g.astype(int)
            label[label == 1] = 0
            label[label == 2] = 1
            self.g = label

        # self.transforms = transforms.Compose([
        #                     transforms.RandomCrop(opt.input_size_path),
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        '''
        self.transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomCrop(opt.input_size_path),
                            # transforms.Resize([opt.input_size_path, opt.input_size_path]),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        '''
        if split =='test':
            self.transforms = self.transforms_test
        else:
            self.transforms = self.transforms_train

    def transforms_train(self, image, sp_mask): 
        
        #slic = Slic(num_components=self.opt.num_superpixels, compactness=10)
        #sp_mask = slic.iterate(np.array(image)) #superpixel mask 
        #sp_mask = Image.fromarray(sp_mask)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            sp_mask = TF.hflip(sp_mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            sp_mask = TF.vflip(sp_mask)            
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.opt.input_size_path, self.opt.input_size_path))
        image = TF.crop(image, i, j, h, w)
        sp_mask = TF.crop(sp_mask, i, j, h, w)            
        
        # Random ColorJitter
        colorjitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01)
        colorjitter = get_random_color_jitter(brightness=colorjitter.brightness, contrast=colorjitter.contrast, saturation=colorjitter.saturation, hue=colorjitter.hue) # if need to enhance multiple images
        image = colorjitter(image)
        
        # Resize
        #resize = transforms.Resize(size=(520, 520))
        #image = resize(image)
        #sp_mask = resize(sp_mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        sp_mask = torch.tensor(np.array(sp_mask)).type(torch.LongTensor)
              
        return image, sp_mask    

    def transforms_test(self, image, sp_mask): 
        
        #slic = Slic(num_components=self.opt.num_superpixels, compactness=10)
        #sp_mask = slic.iterate(np.array(image)) #superpixel mask 
        #sp_mask = Image.fromarray(sp_mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.opt.input_size_path, self.opt.input_size_path))
        image = TF.crop(image, i, j, h, w)
        sp_mask = TF.crop(sp_mask, i, j, h, w)            

        # Transform to tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        sp_mask = torch.tensor(np.array(sp_mask)).type(torch.LongTensor)
        
        return image, sp_mask

    def __getitem__(self, index):
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
        single_g = torch.tensor(self.g[index]).type(torch.LongTensor)

        if self.mode == "path" or self.mode == 'pathpath':
            #meilu
            if getpass.getuser() == 'meiluzhu2':
                image_path = '/home/meiluzhu2/data/MIM'+ self.X_path[index].split('data')[-1]
                single_X_path = Image.open(image_path).convert('RGB')
                sp_mask = self.superpixels[self.X_path[index]]
                
            else:
                single_X_path = Image.open(self.dataroot + self.X_path[index]).convert('RGB')
                sp_mask = self.superpixels[self.X_path[index]]
                
            # print(single_X_path, self.transforms(single_X_path).shape)
            #if self.split =='test':
            #    image, sp_mask = self.transform_test(single_X_path)
            #else:
            #    image, sp_mask = self.transform_train(single_X_path)
                
            return (self.transforms(single_X_path, sp_mask), 0, 0, single_e, single_t, single_g)
        
        elif self.mode == "omic" or self.mode == 'omicomic':
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (0, 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathomic":                    
            #meilu
            if getpass.getuser() == 'meiluzhu2':
                image_path = '/home/meiluzhu2/data/MIM'+ self.X_path[index].split('data')[-1]
                single_X_path = Image.open(image_path).convert('RGB')
                sp_mask = self.superpixels[self.X_path[index]]
                single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            else:
                single_X_path = Image.open(self.dataroot + self.X_path[index]).convert('RGB')
                sp_mask = self.superpixels[self.X_path[index]]
                single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)                           
            
            return (self.transforms(single_X_path, sp_mask), 0, single_X_omic, single_e, single_t, single_g)

    def __len__(self):
        return len(self.X_path)


#################################
# Dataloader with memory bank
#################################
class Pathomic_InstanceSample(Dataset):
    def __init__(self, opt, data, split, mode='omic', is_sample=True):
        super(Pathomic_InstanceSample, self).__init__()

        self.p = opt.nce_p
        self.k = opt.nce_k
        self.is_sample = is_sample
        self.pos_mode = opt.pos_mode

        self.X_path = data[split]['x_path']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.g = data[split]['g']
        self.mode = mode
        self.task = opt.task
        self.opt = opt
        self.superpixels = {}
        self.dataroot = opt.dataroot
        for i in range(len(self.X_path)):
            self.X_path[i] = self.X_path[i].replace("./data/TCGA_GBMLGG/", "/")        

        '''
        self.transforms = TransformTwice(transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomCrop(opt.input_size_path),
                            # transforms.Resize([opt.input_size_path, opt.input_size_path]),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        '''

        self.num_samples = len(self.X_path)
        # generate superpixel
        if self.mode == "path" or self.mode == 'pathpath' or self.mode == 'pathomic':
            #meilu
            for index in range(len(self.X_path)):
                if getpass.getuser() == 'meiluzhu2':
                    image_path = '/home/meiluzhu2/data/MIM'+ self.X_path[index].split('data')[-1]
                    single_X_path = Image.open(image_path).convert('RGB')
                    slic = Slic(num_components=opt.num_superpixels, compactness=10)
                    sp_mask = slic.iterate(np.array(single_X_path)) #superpixel mask 
                    sp_mask = Image.fromarray(sp_mask)                    
                    self.superpixels[self.X_path[index]] = sp_mask
                else:
                    single_X_path = Image.open(self.dataroot + self.X_path[index]).convert('RGB') 
                    slic = Slic(num_components=opt.num_superpixels, compactness=10)
                    sp_mask = slic.iterate(np.array(single_X_path)) #superpixel mask 
                    sp_mask = Image.fromarray(sp_mask)                    
                    self.superpixels[self.X_path[index]] = sp_mask    
            print('The number of superpixel:',len(self.superpixels))


        if opt.task == "grad":
            ## newly added, 2021/12/22
            if opt.label_dim == 3:
                num_classes = 3
                label = self.g.astype(int)
            
            elif opt.label_dim == 2:
                ### 改成二分类，将标签中的1改为0, 标签中的2改为1
                num_classes = 2
                label = self.g.astype(int)
                print("original labels:", label)
                label[label == 1] = 0
                label[label == 2] = 1
                print("modified labels:", label)
                self.g = label

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(self.num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)
            
            # print("positive:", self.cls_positive)
            # print("negative:", self.cls_negative)

    def transforms(self, image, sp_mask): 
                    
        #slic = Slic(num_components=self.opt.num_superpixels, compactness=10)
        #sp_mask = slic.iterate(np.array(image)) #superpixel mask 
        #sp_mask = Image.fromarray(sp_mask)
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            sp_mask = TF.hflip(sp_mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            sp_mask = TF.vflip(sp_mask)            
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.opt.input_size_path, self.opt.input_size_path))
        image = TF.crop(image, i, j, h, w)
        sp_mask = TF.crop(sp_mask, i, j, h, w)            
        
        # Random ColorJitter
        colorjitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01)
        colorjitter = get_random_color_jitter(brightness=colorjitter.brightness, contrast=colorjitter.contrast, saturation=colorjitter.saturation, hue=colorjitter.hue) # if need to enhance multiple images
        image = colorjitter(image)
        
        # Resize
        #resize = transforms.Resize(size=(520, 520))
        #image = resize(image)
        #sp_mask = resize(sp_mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        sp_mask = torch.tensor(np.array(sp_mask)).type(torch.LongTensor)

        return image, sp_mask



    def __getitem__(self, index):

        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
        single_g = torch.tensor(self.g[index]).type(torch.LongTensor)
        
        #meilu
        if getpass.getuser() == 'meiluzhu2':
            image_path = '/home/meiluzhu2/data/MIM'+ self.X_path[index].split('data')[-1]
            single_X_path = Image.open(image_path).convert('RGB')
            sp_mask = self.superpixels[self.X_path[index]]
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
        else:
            single_X_path = Image.open(self.dataroot + self.X_path[index]).convert('RGB')
            sp_mask = self.superpixels[self.X_path[index]]
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
        # return (self.transforms(single_X_path), 0, single_X_omic, single_e, single_t, single_g)

        # print("sample index and label:", index, single_g)

        if self.task == "surv":
            pos_idx = index
            all_neg_idx = list(range(0, self.num_samples))
            all_neg_idx.remove(index)
            replace = True if self.k > len(all_neg_idx) else False
            neg_idx = np.random.choice(all_neg_idx, self.k, replace=replace)

        elif self.task == "grad":
            if self.pos_mode == 'exact':
                pos_idx = np.asarray([index])
            elif self.pos_mode == 'relax':
                pos_idx = np.asarray([np.random.choice(self.cls_positive[single_g], 1)[0]])
                # print("anchor:", index, "pos_idx:", pos_idx)
            elif self.pos_mode == 'multi_pos':
                # print("==============multiple positive pairs===============")
                # print("total number of positive samples:", self.cls_positive[single_g].shape)
                pos_idx = np.random.choice(self.cls_positive[single_g], self.p, replace=False)
                pos_idx[0] = index ### make sure the sample is selected as positive pair.
            else:
                raise NotImplementedError(self.pos_mode)

            replace = True if self.k > len(self.cls_negative[single_g]) else False
            neg_idx = np.random.choice(self.cls_negative[single_g], self.k, replace=replace)

        # print("sample index:", index)
        # print("positive index:", pos_idx)
        # print("negative index:", neg_idx)
        sample_idx = np.hstack((pos_idx, neg_idx))

        # print("sample label:", single_g)
        # for i in range(len(sample_idx)):
        #     idx = sample_idx[i]
        #     print("contrast sample label:", self.g[idx])
        
        image, sp_mask = self.transforms(single_X_path,sp_mask)
        image_ema, sp_mask_ema = self.transforms(single_X_path, sp_mask)
        
        image_ema_maksing_v1, _ = self.transforms(single_X_path, sp_mask)
        image_ema_maksing_v2, _ = self.transforms(single_X_path, sp_mask)
        

        return ((image, sp_mask, image_ema, sp_mask_ema, image_ema_maksing_v1, image_ema_maksing_v2), 0, single_X_omic, single_e, single_t, single_g, index, sample_idx)


    def __len__(self):
        return len(self.X_path)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_random_color_jitter(brightness,contrast,saturation,hue):

    transforms_list = []

    if brightness is not None:
        brightness_factor = random.uniform(brightness[0], brightness[1])
        transforms_list.append(transforms.Lambda(lambda img: TF.adjust_brightness(img, brightness_factor)))

    if contrast is not None:
        contrast_factor = random.uniform(contrast[0], contrast[1])
        transforms_list.append(transforms.Lambda(lambda img: TF.adjust_contrast(img, contrast_factor)))

    if saturation is not None:
        saturation_factor = random.uniform(saturation[0], saturation[1])
        transforms_list.append(transforms.Lambda(lambda img: TF.adjust_saturation(img, saturation_factor)))

    if hue is not None:
        hue_factor = random.uniform(hue[0], hue[1])
        transforms_list.append(transforms.Lambda(lambda img: TF.adjust_hue(img, hue_factor)))

    random.shuffle(transforms_list)
    transform = transforms.Compose(transforms_list)

    return transform