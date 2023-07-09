"""
用fixed fuse teacher和病理模型的self EMA teacher对病理单模态模型进行蒸馏，
加载蒸馏完的病理单模态模型，测试效果如何。
"""

import os
import logging
import numpy as np
import random
import pickle

import torch

# Env
from networks_new import define_net
# from data_loaders import *
from options import parse_args
# from train_test_fusion import test
from train_test_path_multi_distill import test_model
# from train_test_path_distill import test_model
# from data_loaders_new import pathomic_dataloader
from data_loaders_MT import pathomic_dataloader, pathomic_patches_dataloader

### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)): 
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

### 2. Initializes Data
ignore_missing_histype = 1 if 'grad' in opt.task else 0
ignore_missing_moltype = 1 if 'omic' in opt.mode else 0
use_patch, roi_dir = '_patch_', 'all_st_patches_512'
# use_patch, roi_dir = '_', 'all_st'
use_rnaseq = '_rnaseq' if opt.use_rnaseq else ''

data_cv_path = '%s/splits_5cv_2022/gbmlgg5cv_all_st_patches_512_1_1_0.pkl' % (opt.dataroot)
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
results, results_path, results_omic = [], [], []
rocauc_fuse_all, ap_fuse_all, f1_micro_fuse_all, f1_gradeIV_fuse_all = [], [], [], []
rocauc_path_all, ap_path_all, f1_micro_path_all, f1_gradeIV_path_all = [], [], [], []
rocauc_omic_all, ap_omic_all, f1_micro_omic_all, f1_gradeIV_omic_all = [], [], [], []

### 3. Sets-Up Main Loop
for k, data in data_cv_splits.items():
    if k > 0:
        print("*******************************************")
        print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
        print("*******************************************")
        ### 3.2 Evalutes Train + Test Error, and Saves Model
        train_loader, test_loader, n_data = pathomic_dataloader(opt, data)
        avg_all_metrics = np.array([0.0, 0.0, 0.0, 0.0])

        for epoch in range(opt.niter_decay, opt.niter_decay+1):
        # for which_model in ['_best', '']:
            load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, \
                opt.model_name, '%s_%d_best.pt' % (opt.model_name, k))
            model_ckpt = torch.load(load_path, map_location=device)

            #### Loading Env
            model_state_dict = model_ckpt['model_state_dict']
            if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata

            model = define_net(opt, None)
            if isinstance(model, torch.nn.DataParallel): model = model.module

            print('Loading the model from %s' % load_path)
            model.load_state_dict(model_state_dict)
            model = torch.nn.DataParallel(model)

            loss_test, cindex_test, pvalue_test, _, grad_acc_test, all_grad_metrics, pred_test, \
                _, feats_test = test_model(opt, model, test_loader, device)
            # print(feats_test.shape)
            # pickle.dump(feats_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, \
            #     opt.model_name, '%s_%d%stest_features_epoch%d.pkl' % (opt.model_name, k, use_patch, epoch)), 'wb'))
            
            # save_path = os.path.join(opt.checkpoints_dir, opt.exp_name, "feature_analysis_5cv_20220223")
            # print("saving the model predictions and gradients at:", save_path)
            # pickle.dump(feats_test, open(os.path.join(save_path, '%s_%d%stest_features.pkl' % (
            #     opt.model_name, k, use_patch)), 'wb'))

            # preds_path, feats_path, gt_all = pred_test[6], feats_test[1], feats_test[-1]
            # # print(preds_path.shape, feats_path.shape, gt_all.shape)
            # preds_feats_test = [preds_path, feats_path, gt_all]
            # pickle.dump(preds_feats_test, open(os.path.join(save_path, '%s_%d%spreds_feats_test.pkl' % (
            #     opt.model_name, k, use_patch)), 'wb'))

            avg_all_metrics += np.array(all_grad_metrics)

        all_grad_metrics = avg_all_metrics/1.0
        # all_grad_metrics = avg_all_metrics
        
        if opt.task == 'surv':
            print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
            results.append(cindex_test)

        elif opt.task == 'grad':
            print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
            results.append(grad_acc_test)
            [rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path] = all_grad_metrics

            rocauc_path_all.append(rocauc_path)
            ap_path_all.append(ap_path)
            f1_micro_path_all.append(f1_micro_path)
            f1_gradeIV_path_all.append(f1_gradeIV_path)


        # pickle.dump(pred_train, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_train.pkl' % (opt.model_name, k, use_patch)), 'wb'))
        # pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch)), 'wb'))

        if opt.task == 'surv':
            print('Accuracy:', results)
            print("Average:", np.array(results).mean())

        elif opt.task == 'grad':
            # print("[Pathology CNN] AUC: %.4f, AP: %.4f, F1_score: %.4f, F1_GradeIV: %.4f" % (np.array(rocauc_path_all).mean(), \
            #     np.array(ap_path_all).mean(), np.array(f1_micro_path_all).mean(), np.array(f1_gradeIV_path_all).mean()))  
            print("[Pathology CNN] AUC: %.4f +/- %.4f, AP: %.4f +/- %.4f, F1_score: %.4f +/- %.4f, F1_GradeIV: %.4f +/- %.4f" % (
                np.array(rocauc_path_all).mean(), np.array(rocauc_path_all).std(), np.array(ap_path_all).mean(), \
                np.array(ap_path_all).std(), np.array(f1_micro_path_all).mean(), np.array(f1_micro_path_all).std(), \
                np.array(f1_gradeIV_path_all).mean(), np.array(f1_gradeIV_path_all).std()))  