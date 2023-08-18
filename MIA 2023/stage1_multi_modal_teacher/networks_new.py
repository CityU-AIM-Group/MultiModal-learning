"""
Author: Xing Xiaohan
pathomic fusion model that can simultaneously train the two modalities.

update:2021/12/16
For each branch, compute the gradients of hazard over feature vector. 
Return the gradients of each batch, and combine all batches to save.
"""
# Base / Native
import csv
from collections import Counter
import copy
import json
import functools
import gc
import logging
import math
import os
import pdb
import pickle
import random
import sys
import time

# Numerical / Array
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, bilinear
from torch.autograd import Variable
from torch.nn import init, Parameter
from torch.utils.data import DataLoader
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.optim.lr_scheduler as lr_scheduler


# Env
from fusion import *
from options import parse_args
from utils import *
from resnets import ResNet18

# from my_utils.compute_gradients import get_grad_embedding


################
# Network Utils
################
def define_net(opt, k, path_only = False, omic_only = False):
    net = None
    act = define_act_layer(act_type=opt.act_type)
    init_max = True if opt.init_type == "max" else False

    if opt.mode == "path":
        net = get_resnet(path_dim=opt.path_dim, act=act, label_dim=opt.label_dim)
    elif opt.mode == "omic":
        net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, \
            act=act, label_dim=opt.label_dim, init_max=init_max)
    elif opt.mode == "pathomic":        
        if not path_only and not omic_only:
            if opt.fusion_type == "mmdynamics":
                net = PathomicNet_dynamics(opt=opt, act=act, k=k)
            else:
                net = PathomicNet(opt=opt, act=act, k=k)
        if path_only:
            # print("creating path only model")
            net = get_resnet(path_dim=opt.path_dim, act=act, label_dim=opt.label_dim)
        if omic_only:
            net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, \
                act=act, label_dim=opt.label_dim, init_max=init_max)
    else:
        raise NotImplementedError('model [%s] is not implemented' % opt.model)
    return init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)

def define_decoder_net(opt, k):

    act = define_act_layer(act_type=opt.act_type)
    init_max = True if opt.init_type == "max" else False    
    path_decoder = ResnetDecoder(latent_size = opt.path_dim)
    omic_decoder = MaxNetDecoder(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, init_max=init_max)
    
    return path_decoder, omic_decoder

def define_optimizer(opt, model):
    optimizer = None
    if opt.optimizer_type == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=opt.final_lr)
    elif opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, initial_accumulator_value=0.1)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer


def define_reg(opt, model):
    loss_reg = None
    
    if opt.reg_type == 'none':
        loss_reg = 0
    elif opt.reg_type == 'path':
        loss_reg = regularize_path_weights(model=model)
    elif opt.reg_type == 'mm':
        loss_reg = regularize_MM_weights(model=model)
    elif opt.reg_type == 'all':
        loss_reg = regularize_weights(model=model)
    elif opt.reg_type == 'omic':
        loss_reg = regularize_MM_omic(model=model)
    else:
        raise NotImplementedError('reg method [%s] is not implemented' % opt.reg_type)
    return loss_reg


def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
       scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
       scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
       scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    elif opt.lr_policy == 'onecycle':
       scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=opt.niter+opt.niter_decay, steps_per_epoch=200)
    else:
       return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer


def define_bifusion(fusion_type, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=32, dim2=32, scale_dim1=1, scale_dim2=1, mmhid=32, dropout_rate=0.25):
    fusion = None
    if fusion_type == 'pofusion':
        fusion = BilinearFusion(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, dim1=dim1, dim2=dim2, scale_dim1=scale_dim1, scale_dim2=scale_dim2, mmhid=mmhid, dropout_rate=dropout_rate)
    elif fusion_type == 'concat':
        fusion = None
    else:
        raise NotImplementedError('fusion type [%s] is not found' % fusion_type)
    return fusion


def define_LMF_bifusion(opt):
    """
    low-rank multi-modal fusion.
    """
    rank = 4
    hid_dim = 16
    dropout_rate = opt.dropout_rate
    fusion = LMF_bifusion((opt.path_dim, opt.omic_dim), (hid_dim, hid_dim), 
                          (dropout_rate, dropout_rate, 0.5), opt.mmhid, rank)
    return fusion


def define_HFB_bifusion(opt):
    rank = 20
    hid_dim = opt.mmhid
    dropout_rate = opt.dropout_rate
    fusion = HFB_fusion((opt.omic_dim, opt.path_dim), (hid_dim, hid_dim), (opt.mmhid, opt.mmhid), 
                        (dropout_rate, dropout_rate), rank, 0.1)
    return fusion



############
# Omic Model
############
class MaxNet(nn.Module):
    def __init__(self, input_dim=80, omic_dim=32, return_grad = 'False', dropout_rate=0.25, act=None, label_dim=1, init_max=True):
        super(MaxNet, self).__init__()
        hidden = [64, 48, 32, 32]
        self.act = act
        self.return_grad = return_grad

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            # nn.BatchNorm1d(hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            # nn.BatchNorm1d(hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            # nn.BatchNorm1d(hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            # nn.BatchNorm1d(omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        # self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.relu = nn.ReLU(inplace=False)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))

        if init_max: init_max_weights(self)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        features = self.relu(self.encoder(x))
        out = self.classifier(features)

        if self.return_grad == "True":
            omic_grads = get_grad_embedding(out, features).detach().cpu().numpy()
        else:
            omic_grads = None

        if self.act is not None:
            pred = self.act(out)

            if isinstance(self.act, nn.Sigmoid):
                pred = pred * self.output_range + self.output_shift

        # if self.return_grad == "True":
        #     y_c = torch.sum(out)
        #     features.grad = None
        #     features.retain_grad()
        #     y_c.backward(retain_graph=True)
        #     omic_grads = features.grad.detach().cpu().numpy()
        #     omic_grad_norm = np.linalg.norm(omic_grads, axis=1)
        #     # print("gradient magnitude of the omic feature:", omic_grad_norm)
        #     # print("predicted hazard of the omic modality:", np.reshape(out.detach().cpu().numpy(), (-1)))
        # else:
        #     omic_grads = None

        return features, out, pred, omic_grads



############
# Path Model
############
def get_resnet(path_dim=32, act=None, label_dim=1, **kwargs):
    return ResNet18(path_dim=path_dim, act=act, num_classes=label_dim, **kwargs)



##############################################################################
# Path + Omic  ### train the CNN and SNN models simultaneously.
##############################################################################

class PathomicNet(nn.Module):
    def __init__(self, opt, act, k):
        super(PathomicNet, self).__init__()
        init_max = True if opt.init_type == "max" else False
        self.path_net = get_resnet(path_dim=opt.path_dim, act=act, label_dim=opt.label_dim, return_grad = opt.return_grad)
        self.omic_net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, return_grad = opt.return_grad,
                               dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim, init_max=init_max)
        self.bilinear_dim = 20
        self.task = opt.task

        self.fusion = define_bifusion(fusion_type=opt.fusion_type, skip=opt.skip, 
                            use_bilinear=opt.use_bilinear, gate1=opt.path_gate, 
                            gate2=opt.omic_gate, dim1=opt.path_dim, dim2=opt.omic_dim, 
                            scale_dim1=opt.path_scale, scale_dim2=opt.omic_scale, 
                            mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
        
        if opt.fusion_type == "concat":
            self.classifier = nn.Sequential(nn.Linear(opt.mmhid*2, opt.label_dim))
        else:
            self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))

        self.act = act
        self.return_grad = opt.return_grad
        self.cut_fuse_grad = opt.cut_fuse_grad
        self.fusion_type = opt.fusion_type

        # dfs_freeze(self.omic_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        # path_vec = kwargs['x_path']
        path_vec_f3, path_vec, hazard_path, pred_path, path_grads = self.path_net(x_path=kwargs['x_path'])
        omic_vec, hazard_omic, pred_omic, omic_grads = self.omic_net(x_omic=kwargs['x_omic'])
        
        # print("path feature:", torch.mean(path_vec), torch.max(path_vec))
        # print("omic feature:", torch.mean(omic_vec), torch.max(omic_vec))

        if self.cut_fuse_grad:
            if self.fusion_type == "concat":
                features = torch.cat((path_vec.clone().detach(), omic_vec.clone().detach()), 1)
            else:
                features = self.fusion(path_vec.clone().detach(), omic_vec.clone().detach())
        else:
            if self.fusion_type == "concat":
                features = torch.cat((path_vec, omic_vec), 1)
            else:
                features = self.fusion(path_vec, omic_vec)
        
        # print("features for different branches:", path_vec.shape, omic_vec.shape, features.shape)
        hazard = self.classifier(features)
        # print("predictions:", F.softmax(hazard, dim=1))

        if self.return_grad == "True":
            fuse_grads = get_grad_embedding(hazard, features).detach().cpu().numpy()
        else:
            fuse_grads = None
        # print(fuse_grads.shape)
        # print(torch.sum(torch.abs(fuse_grads), axis=1))

        if self.act is not None:
            # print("act", self.act)
            pred = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                pred = pred * self.output_range + self.output_shift

        ### logits for the three branches.
        logits = [hazard_path, hazard_omic, hazard]

        # if self.return_grad == "True":
        #     y_c = torch.sum(hazard)
        #     features.grad = None
        #     features.retain_grad()
        #     y_c.backward(retain_graph=True)
        #     fuse_grads = features.grad.detach().cpu().numpy()
        #     fuse_grad_norm = np.linalg.norm(fuse_grads, axis=1)
        #     # print("gradient magnitude of the fused feature:", fuse_grad_norm)
        #     # print("predicted hazard of the fused branch:", np.reshape(hazard.detach().cpu().numpy(), (-1)))
        
        # else:
        #     fuse_grads = None

        # print("fuse branch:", torch.exp(hazard))
        # print("path branch:", torch.exp(hazard_path))
        # print("omic branch:", torch.exp(hazard_omic))

        # return features.detach().cpu().numpy(), path_vec.detach().cpu().numpy(), omic_vec.detach().cpu().numpy(), \
        #     hazard, hazard_path, hazard_omic, fuse_grads, path_grads, omic_grads
        return features, path_vec, omic_vec, path_vec_f3, logits, pred, pred_path, pred_omic, \
            fuse_grads, path_grads, omic_grads


    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False


class PathomicNet_dynamics(nn.Module):
    def __init__(self, opt, act, k):
        super(PathomicNet_dynamics, self).__init__()
        init_max = True if opt.init_type == "max" else False
        self.path_net = get_resnet(path_dim=opt.path_dim, act=act, label_dim=opt.label_dim, return_grad = opt.return_grad)
        self.omic_net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, return_grad = opt.return_grad,
                               dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim, init_max=init_max)

        dim_list = [opt.path_dim, opt.omic_dim]
        self.fusion = MMDynamic(dim_list, hidden_dim=[opt.mmhid], num_class=opt.label_dim, dropout=0.5)
            
        self.act = act
        self.return_grad = opt.return_grad
        self.cut_fuse_grad = opt.cut_fuse_grad

        # dfs_freeze(self.omic_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, labels, infer, **kwargs):
        # path_vec = kwargs['x_path']
        path_vec_f3, path_vec, hazard_path, pred_path, path_grads = self.path_net(x_path=kwargs['x_path'])
        omic_vec, hazard_omic, pred_omic, omic_grads = self.omic_net(x_omic=kwargs['x_omic'])

        features_list = [path_vec, omic_vec]
        if infer:
            MMlogit = self.fusion.infer(features_list)
            return MMlogit
        else:
            MMLoss, MMlogit, TCPLogit = self.fusion(features_list, labels)
            return MMLoss, MMlogit, TCPLogit


    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False


class ResnetDecoder(nn.Module):
    def __init__(self, latent_size=128, **kwargs):
        super(ResnetDecoder, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 512*2*2, bias=False) # [bs, 512, 2, 2]

        self.Deconv = nn.Sequential(
            nn.ConvTranspose2d(   512,      512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True), # [bs, 512, 4, 4]

            nn.ConvTranspose2d(   512,      256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True), # [bs, 256, 8, 8]

            nn.ConvTranspose2d(   256,      256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),   # [bs, 256, 16, 16]          

            nn.ConvTranspose2d(   256,      128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),  # [bs, 128, 32, 32]

            nn.ConvTranspose2d(   128,      128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),  # [bs, 128, 64, 64]

            nn.ConvTranspose2d(   128,        3, 4, stride=2, padding=1),
            nn.Tanh()                # [bs, 3, 128, 128]
        )

    def forward(self, x):
        batch_size = x.shape[0] # B, N
        
        x = self.fc1(x)
        x = x.resize(batch_size, 512, 2, 2)
        x = self.Deconv(x)

        return x

class MaxNetDecoder(nn.Module):
    def __init__(self, input_dim=80, omic_dim=32, dropout_rate=0.25, act=None, init_max=True):
        super(MaxNetDecoder, self).__init__()
        hidden = [64, 48, 32, 32]

        encoder4 = nn.Sequential(
            nn.Linear(omic_dim, hidden[2]),
            # nn.BatchNorm1d(omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder3 = nn.Sequential(
            nn.Linear(hidden[2], hidden[1]),
            # nn.BatchNorm1d(hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(hidden[1], hidden[0]),
            # nn.BatchNorm1d(hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder1 = nn.Sequential(
            nn.Linear( hidden[0], input_dim))
        
        # self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.encoder = nn.Sequential(encoder4, encoder3, encoder2, encoder1)

        if init_max: init_max_weights(self)


    def forward(self, x):

        x_rec = self.encoder(x)

        return x_rec











