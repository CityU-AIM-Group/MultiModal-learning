import os
import logging
import numpy as np
import random
import pickle
import pandas as pd

import torch

# Env
from data_loaders_MT import pathomic_dataloader, pathomic_patches_dataloader
from networks_new import define_net
from options import parse_args
# from train_test_MT_ensemble import test
from train_test_MT import test
# from train_test_tSVD import test
# from train_test_mmdynamics import test

### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

### 2. Initializes Data
ignore_missing_histype = 1 if 'grad' in opt.task else 0
ignore_missing_moltype = 1 if 'omic' in opt.mode else 0
# ignore_missing_moltype = 0
# use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_vgg_features else ('_', 'all_st')
use_patch, roi_dir = '_patch_', 'all_st_patches_512'
# use_patch, roi_dir = '_', 'all_st'
use_rnaseq = '_rnaseq' if opt.use_rnaseq else ''

# data_cv_path = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
data_cv_path = '%s/splits_5cv_2022/gbmlgg5cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
cindex_fuse, cindex_path, cindex_omic = [], [], []
rocauc_fuse_all, ap_fuse_all, f1_micro_fuse_all, f1_gradeIV_fuse_all = [], [], [], []
rocauc_path_all, ap_path_all, f1_micro_path_all, f1_gradeIV_path_all = [], [], [], []
rocauc_omic_all, ap_omic_all, f1_micro_omic_all, f1_gradeIV_omic_all = [], [], [], []

### 3. Sets-Up Main Loop
for k, data in data_cv_splits.items():
	if k > 0:
		print("*******************************************")
		print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
		print("*******************************************")
		load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k))
		model_ckpt = torch.load(load_path, map_location=device)

		train_loader, test_loader, n_data = pathomic_dataloader(opt, data)		

		#### Loading Env
		model_state_dict = model_ckpt['model_state_dict']
		if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata

		model = define_net(opt, None)
		if isinstance(model, torch.nn.DataParallel): model = model.module

		print('Loading the model from %s' % load_path)
		model.load_state_dict(model_state_dict, strict=False)
		model = torch.nn.DataParallel(model)

		### 3.2 Evalutes Train + Test Error, and Saves Model
		# loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(opt, model, data, 'test', device)
		# _,  _, _, _, _, _, _, _, _, _, _, _, pred_train, grads_train, feats_train = test(opt, model, data, 'train', device)
		loss_test, _, _, _, cindex_test, cindex_test_path, cindex_test_omic, pvalue_test, surv_acc_test, grad_acc_test, \
			grad_path_test, grad_omic_test, all_grad_metrics, pred_test, grads_test, feats_test = test(opt, model, model, test_loader, device)
		# grad_acc_test, grad_path_test, grad_omic_test, all_grad_metrics = test(
		# 	opt, model, model, test_loader, device)

		# print("model predictions:", pred_test)
		# print("feature gradients:", grads_test)
		## pred_test: [risk_pred_all, risk_path_all, risk_omic_all, survtime_all, censor_all, probs_all, gt_all]
				
		# save_path = "./feature_analysis_5cv_20221225"
		# print("saving the model predictions and gradients at:", save_path)

		# preds_fuse, feats_fuse, gt_all = pred_test[5], feats_test[0], feats_test[-1]
		# # print(preds_path.shape, feats_path.shape, gt_all.shape)
		# preds_feats_test = [preds_fuse, feats_fuse, gt_all]
		# pickle.dump(preds_feats_test, open(os.path.join(save_path, '%s_%d%spreds_feats_test.pkl' % (
		# 	opt.model_name, k, use_patch)), 'wb'))

		# [feat_fuse_all, feat_path_all, feat_omic_all, gt_all] = feats_test
		# print(feat_path_all.shape, feat_omic_all.shape, gt_all)
		# feats_labels = np.concatenate((feat_path_all, feat_omic_all, np.reshape(gt_all+1, (-1,1))), 1)
		# print("feats_labels:", feats_labels.shape)
		# root_path = "/home/xiaoxing/xxh_codes/pathomic_fusion_SOTA_methods_20220214/MMGL-main/MMGL/data/GBMLGG"
		# pd.DataFrame(feats_labels).to_csv(root_path + "/split" + str(k) + "_train.csv", index=False)


		if opt.task == 'surv':
			# print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
			print("[Final] Apply model to testing set: C-Index: %.10f, pathology: %.10f, genomics: %.10f, \
				P-Value: %.10e" % (cindex_test, cindex_test_path, cindex_test_omic, pvalue_test))
			logging.info("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
			cindex_fuse.append(cindex_test)
			cindex_path.append(cindex_test_path)
			cindex_omic.append(cindex_test_omic)

		elif opt.task == 'grad':
			# print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f, pathology: %.4f, genomics: %.4f" % (
			# 	loss_test, grad_acc_test, grad_path_test, grad_omic_test))
			# logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
			# results.append(grad_acc_test)

			[rocauc_fuse, ap_fuse, f1_micro_fuse, f1_gradeIV_fuse, rocauc_path, ap_path, f1_micro_path, \
				f1_gradeIV_path, rocauc_omic, ap_omic, f1_micro_omic, f1_gradeIV_omic] = all_grad_metrics
			
			rocauc_fuse_all.append(rocauc_fuse)
			ap_fuse_all.append(ap_fuse)
			f1_micro_fuse_all.append(f1_micro_fuse)
			f1_gradeIV_fuse_all.append(f1_gradeIV_fuse)

			rocauc_path_all.append(rocauc_path)
			ap_path_all.append(ap_path)
			f1_micro_path_all.append(f1_micro_path)
			f1_gradeIV_path_all.append(f1_gradeIV_path)

			rocauc_omic_all.append(rocauc_omic)
			ap_omic_all.append(ap_omic)
			f1_micro_omic_all.append(f1_micro_omic)
			f1_gradeIV_omic_all.append(f1_gradeIV_omic)


		# # ## 3.3 Saves Model
		# result_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch))
		# pickle.dump(pred_test, open(result_path, 'wb'))

		# save_path = os.path.join(opt.checkpoints_dir, opt.exp_name, "pathomic_feature_visualization")
		# # print("saving the model predictions and gradients at:", save_path)
		# # # pickle.dump(pred_train, open(os.path.join(save_path, '%s_%d%spred_train.pkl' % (opt.model_name, k, use_patch)), 'wb'))
		# # # pickle.dump(grads_train, open(os.path.join(save_path, '%s_%d%sgrad_train.pkl' % (opt.model_name, k, use_patch)), 'wb'))
		# # # pickle.dump(feats_train, open(os.path.join(save_path, '%s_%d%sfeat_train.pkl' % (opt.model_name, k, use_patch)), 'wb'))
		# # pickle.dump(pred_test, open(os.path.join(save_path, '%s_%d%spred_train.pkl' % (opt.model_name, k, use_patch)), 'wb'))
		# # pickle.dump(grads_test, open(os.path.join(save_path, '%s_%d%sgrad_train.pkl' % (opt.model_name, k, use_patch)), 'wb'))
		# pickle.dump(feats_test, open(os.path.join(save_path, '%s_%d%sfeats_test.pkl' % (opt.model_name, k, use_patch)), 'wb'))


		# print('Split Results:', results)
		# print("Average:", np.array(results).mean())

		if opt.task == 'grad':
			# print("[Pathomic Fuse] AUC: %.4f, AP: %.4f, F1_score: %.4f, F1_GradeIV: %.4f" % (np.array(rocauc_fuse_all).mean(), \
			# 	np.array(ap_fuse_all).mean(), np.array(f1_micro_fuse_all).mean(), np.array(f1_gradeIV_fuse_all).mean()))
			print("[Pathomic Fuse] AUC: %.4f +/- %.4f, AP: %.4f +/- %.4f, F1_score: %.4f +/- %.4f, F1_GradeIV: %.4f +/- %.4f" % (
					np.array(rocauc_fuse_all).mean(), np.array(rocauc_fuse_all).std(), np.array(ap_fuse_all).mean(), \
					np.array(ap_fuse_all).std(), np.array(f1_micro_fuse_all).mean(), np.array(f1_micro_fuse_all).std(), \
					np.array(f1_gradeIV_fuse_all).mean(), np.array(f1_gradeIV_fuse_all).std()))  
			print("[Pathology CNN] AUC: %.4f, AP: %.4f, F1_score: %.4f, F1_GradeIV: %.4f" % (np.array(rocauc_path_all).mean(), \
				np.array(ap_path_all).mean(), np.array(f1_micro_path_all).mean(), np.array(f1_gradeIV_path_all).mean()))
			print("[Genomics SNN] AUC: %.4f, AP: %.4f, F1_score: %.4f, F1_GradeIV: %.4f" % (np.array(rocauc_omic_all).mean(), \
				np.array(ap_omic_all).mean(), np.array(f1_micro_omic_all).mean(), np.array(f1_gradeIV_omic_all).mean()))

		elif opt.task == 'surv':
				print("[Pathomic Fuse] c-index: %.4f +/- %.4f" % (np.array(cindex_fuse).mean(), np.array(cindex_fuse).std()))
				print("[Pathology] c-index: %.4f +/- %.4f" % (np.array(cindex_path).mean(), np.array(cindex_path).std()))
				print("[Genomics] c-index: %.4f +/- %.4f" % (np.array(cindex_omic).mean(), np.array(cindex_omic).std()))  

# pickle.dump(results, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))