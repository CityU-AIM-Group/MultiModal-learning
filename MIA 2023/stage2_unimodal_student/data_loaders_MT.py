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


def pathomic_dataloader(opt, data):

    custom_data_loader = Pathomic_InstanceSample(opt, data, split='train', mode=opt.mode)
    print("number of training samples:", len(custom_data_loader))
    train_class_idx = custom_data_loader.cls_positive ### 每个类别的样本index
    train_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader, batch_size=opt.batch_size, 
        num_workers=4, shuffle=True, collate_fn=mixed_collate, drop_last=True)
    
    n_data = len(custom_data_loader)

    test_data_loader = PathomicDatasetLoader(opt, data, split='test', mode=opt.mode)
    print("number of testing samples:", len(test_data_loader))
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data_loader, batch_size=opt.batch_size, 
        num_workers=4, shuffle=False, collate_fn=mixed_collate)

    return train_loader, train_class_idx, test_loader, n_data



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
        # if split == "test":
        #     print(data[split]['x_path'])
        self.X_path = data[split]['x_path']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.g = data[split]['g']
        self.mode = mode
        self.dataroot = opt.dataroot
        for i in range(len(self.X_path)):
            self.X_path[i] = self.X_path[i].replace("./data/TCGA_GBMLGG/", "/")

        ### 2022/12/27, xxh
        ### 之前是测试时也有transform, 现在改成只有训练时有transform. 
        if split == 'test':
            self.transforms = transforms.Compose([transforms.RandomCrop(opt.input_size_path), transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        else:
            self.transforms = transforms.Compose([
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomVerticalFlip(0.5),
                                transforms.RandomCrop(opt.input_size_path),
                                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
        single_g = torch.tensor(self.g[index]).type(torch.LongTensor)

        if self.mode == "path" or self.mode == 'pathpath':
            single_X_path = Image.open(self.dataroot + self.X_path[index]).convert('RGB')
            # print(single_X_path, self.transforms(single_X_path).shape)
            return (self.transforms(single_X_path), 0, 0, single_e, single_t, single_g)
        elif self.mode == "omic" or self.mode == 'omicomic':
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (0, 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathomic":
            single_X_path = Image.open(self.dataroot + self.X_path[index]).convert('RGB')
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            # print(single_X_path, self.transforms(single_X_path).shape)
            return (self.transforms(single_X_path), 0, single_X_omic, single_e, single_t, single_g)

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
        self.neg_mode = opt.neg_mode
        self.distill = opt.distill

        # print("data split:", data[split])

        self.X_path = data[split]['x_path']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.g = data[split]['g']
        self.mode = mode
        self.task = opt.task
        self.dataroot = opt.dataroot
        for i in range(len(self.X_path)):
            self.X_path[i] = self.X_path[i].replace("./data/TCGA_GBMLGG/", "/")

        # print("grading labels:", self.g)
        
        self.transforms = TransformTwice(transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomCrop(opt.input_size_path),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        self.num_samples = len(self.X_path)

        if opt.task == "grad":
            ## newly added, 2021/12/22
            if opt.label_dim == 3:
                num_classes = 3
                label = self.g.astype(int)

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
            
            # print("gt labels:", label)
            # print("positive:", self.cls_positive)
            # print("negative:", self.cls_negative)

    def __getitem__(self, index):

        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
        single_g = torch.tensor(self.g[index]).type(torch.LongTensor)

        single_X_path = Image.open(self.dataroot + self.X_path[index]).convert('RGB')
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

            if self.neg_mode == "all_others":
                ### 把除了query之外的所有的样本作为negative pairs. 
                all_neg_idx = list(range(0, self.num_samples))
                all_neg_idx.remove(index)
                replace = True if self.k > len(all_neg_idx) else False
                neg_idx = np.random.choice(all_neg_idx, self.k, replace=replace)
            elif self.neg_mode == "diff_class":
                ### 把不同类别的样本作为negative pairs. 
                replace = True if self.k > len(self.cls_negative[single_g]) else False
                neg_idx = np.random.choice(self.cls_negative[single_g], self.k, replace=replace)
            elif self.neg_mode == "both_models":
                ### 从对方模型和自己模型中都选择负样本对, 第一部分是对方模型除了query之外的所有其他样本都作为negative pairs.
                all_neg_idx = list(range(0, self.num_samples))
                all_neg_idx.remove(index)
                replace = True if self.k > len(all_neg_idx) else False
                neg_idx1 = np.random.choice(all_neg_idx, self.k, replace=replace)
                ### 第二部分是把自己模型中不同类别的样本作为negative pairs. 
                replace = True if self.k > len(self.cls_negative[single_g]) else False
                neg_idx2 = np.random.choice(self.cls_negative[single_g], self.k, replace=replace)
                neg_idx = np.hstack((neg_idx1, neg_idx2))  

        if self.distill == 'hkd':
            sample_idx = neg_idx
        else:
            sample_idx = np.hstack((pos_idx, neg_idx))  

        # print("query index:", index)
        # print("contrast index:", sample_idx.shape)
        # print("query index and label:", index, single_g)
        # for i in range(len(sample_idx)):
        #     idx = sample_idx[i]
        #     # print("contrast sample index:", idx)
        #     print("contrast sample label:", self.g[idx])

        return (self.transforms(single_X_path), 0, single_X_omic, single_e, single_t, single_g, index, sample_idx)


    def __len__(self):
        return len(self.X_path)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        # print("out1:", out1)
        # print("out2:", out2)
        return out1, out2
