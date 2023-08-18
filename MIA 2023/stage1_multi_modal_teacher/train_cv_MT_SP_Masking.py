import os
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'

import logging
import numpy as np
import random
import pickle

import torch

# Env
from data_loaders_MT_SP import pathomic_dataloader, pathomic_patches_dataloader
from options import parse_args
from train_test_MT_SP_Masking import train, test


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
use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_vgg_features else ('_', 'all_st')
use_rnaseq = '_rnaseq' if opt.use_rnaseq else ''

# data_cv_path = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
data_cv_path = '%s/splits_5cv_2022/gbmlgg5cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
results, results_path, results_omic = [], [], []
rocauc_fuse_all, ap_fuse_all, f1_micro_fuse_all, f1_gradeIV_fuse_all, f1_macro_fuse_all, recall_macro_fuse_all, precision_macro_fuse_all, kappa_fuse_all, mcc_fuse_all = [], [], [], [], [], [], [], [], []
rocauc_path_all, ap_path_all, f1_micro_path_all, f1_gradeIV_path_all, f1_macro_path_all, recall_macro_path_all, precision_macro_path_all, kappa_path_all, mcc_path_all = [], [], [], [], [], [], [], [], []
rocauc_omic_all, ap_omic_all, f1_micro_omic_all, f1_gradeIV_omic_all, f1_macro_omic_all, recall_macro_omic_all, precision_macro_omic_all, kappa_omic_all, mcc_omic_all = [], [], [], [], [], [], [], [], []

acc_fuse_all = []
acc_path_all = []
acc_omic_all = []

### 读取裁剪之后的每张ROI对应的9个patches.
roi_dir = 'all_st_patches_512'
# data_cv_path_patches = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
data_cv_path_patches = '%s/splits_5cv_2022/gbmlgg5cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
print("Loading %s" % data_cv_path_patches)
data_cv_patches = pickle.load(open(data_cv_path_patches, 'rb'))
data_cv_splits_patches = data_cv_patches['cv_splits']


### 3. Sets-Up Main Loop
for k, data in data_cv_splits.items():
	# opt.mu = 1e-5
	if k > 0:
		print("*******************************************")
		print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
		print("*******************************************")
		if os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d_patch_pred_train.pkl' % (opt.model_name, k))):
			print("Train-Test Split already made. The model will be overwritten")
			# continue

		train_loader, test_loader, n_data = pathomic_dataloader(opt, data)
		data_patches = data_cv_splits_patches[k]
		test_loader_patches = pathomic_patches_dataloader(opt, data_patches)

		### 3.1 Trains Model
		module_list, model, ema_model, optimizer, metric_logger = train(
			opt, train_loader, n_data, test_loader, test_loader_patches, device, k)

		### 3.2 Evalutes Train + Test Error, and Saves Model
		loss_test, _, _, _, cindex_test, cindex_test_path, cindex_test_omic, pvalue_test, surv_acc_test, grad_acc_test, \
			grad_path_test, grad_omic_test, all_grad_metrics, pred_test, grads_test, feats_test = test(opt, module_list, model, test_loader_patches, device)

		if opt.task == 'surv':
			print("[Final] Apply model to testing set: C-Index: %.10f, pathology: %.10f, genomics: %.10f, \
				P-Value: %.10e" % (cindex_test, cindex_test_path, cindex_test_omic, pvalue_test))
			logging.info("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
			results.append(cindex_test)
			results_path.append(cindex_test_path)
			results_omic.append(cindex_test_omic)

		elif opt.task == 'grad':
			print("[Final] Apply model to testing set: Loss: %.10f, fused Acc: %.4f, pathology: %.4f, genomics: %.4f" % (
				loss_test, grad_acc_test, grad_path_test, grad_omic_test))
			logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
			results.append(grad_acc_test)
			results_path.append(grad_path_test)
			results_omic.append(grad_omic_test)

			[rocauc_fuse, ap_fuse, f1_micro_fuse, f1_gradeIV_fuse,f1_macro_fuse, recall_macro_fuse, precision_macro_fuse, kappa_fuse, mcc_fuse, \
						rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path,f1_macro_path, recall_macro_path, precision_macro_path, kappa_path, mcc_path, \
						rocauc_omic, ap_omic, f1_micro_omic, f1_gradeIV_omic,f1_macro_omic, recall_macro_omic, precision_macro_omic, kappa_omic, mcc_omic] = all_grad_metrics

			
			rocauc_fuse_all.append(rocauc_fuse)
			ap_fuse_all.append(ap_fuse)
			f1_micro_fuse_all.append(f1_micro_fuse)
			f1_gradeIV_fuse_all.append(f1_gradeIV_fuse)
			f1_macro_fuse_all.append(f1_macro_fuse)
			recall_macro_fuse_all.append(recall_macro_fuse)
			precision_macro_fuse_all.append(precision_macro_fuse)
			kappa_fuse_all.append(kappa_fuse)
			mcc_fuse_all.append(mcc_fuse)
			
			rocauc_path_all.append(rocauc_path)
			ap_path_all.append(ap_path)
			f1_micro_path_all.append(f1_micro_path)
			f1_gradeIV_path_all.append(f1_gradeIV_path)
			f1_macro_path_all.append(f1_macro_path)
			recall_macro_path_all.append(recall_macro_path)
			precision_macro_path_all.append(precision_macro_path)
			kappa_path_all.append(kappa_path)
			mcc_path_all.append(mcc_path)

			rocauc_omic_all.append(rocauc_omic)
			ap_omic_all.append(ap_omic)
			f1_micro_omic_all.append(f1_micro_omic)
			f1_gradeIV_omic_all.append(f1_gradeIV_omic)
			f1_macro_omic_all.append(f1_macro_omic)
			recall_macro_omic_all.append(recall_macro_omic)
			precision_macro_omic_all.append(precision_macro_omic)
			kappa_omic_all.append(kappa_omic)
			mcc_omic_all.append(mcc_omic)            
        
		### 3.3 Saves Model
		if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
			model_state_dict = model.module.cpu().state_dict()
			ema_model_state_dict = ema_model.module.cpu().state_dict()
		else:
			model_state_dict = model.cpu().state_dict()
			ema_model_state_dict = ema_model.cpu().state_dict()
		
		save_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k))
		'''
        torch.save({
			'split':k,
			'opt': opt,
			'epoch': opt.niter+opt.niter_decay,
			'data': data,
			'model_state_dict': model_state_dict,
			'ema_model_state_dict': ema_model_state_dict,
			# 'model_state_dict': module_list.cpu().state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'metrics': metric_logger}, 
			save_path)
        '''
		print("saving the model at:", save_path)

		# pickle.dump(pred_train, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_train.pkl' % (opt.model_name, k, use_patch)), 'wb'))
		# pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch)), 'wb'))

		if opt.task == 'surv':
			print('Fusion Results:', results)
			print("Average:", np.array(results).mean())
			print('Pathology Results:', results_path)
			print("Average:", np.array(results_path).mean())
			print('Genomics Results:', results_omic)
			print("Average:", np.array(results_omic).mean())

		elif opt.task == 'grad':
			print("[Pathomic Fuse] AUC: %.4f[%.4f] , AP: %.4f[%.4f], F1_score: %.4f[%.4f], F1_GradeIV: %.4f[%.4f]" % (np.array(rocauc_fuse_all).mean(),np.array(rocauc_fuse_all).std(), \
				np.array(ap_fuse_all).mean(),np.array(ap_fuse_all).std(), np.array(f1_micro_fuse_all).mean(),np.array(f1_micro_fuse_all).std(), np.array(f1_gradeIV_fuse_all).mean(), np.array(f1_gradeIV_fuse_all).std()))
			print("[Pathomic Fuse] F1_macro: %.4f[%.4f] , Recall: %.4f[%.4f], Precision: %.4f[%.4f], Kappa: %.4f[%.4f], MCC: %.4f[%.4f]" % (np.array(f1_macro_fuse_all).mean(),np.array(f1_macro_fuse_all).std(), \
				np.array(recall_macro_fuse_all).mean(),np.array(recall_macro_fuse_all).std(), np.array(precision_macro_fuse_all).mean(),np.array(precision_macro_fuse_all).std(), np.array(kappa_fuse_all).mean(), np.array(kappa_fuse_all).std(), np.array(mcc_fuse_all).mean(), np.array(mcc_fuse_all).std()))

			print("[Pathology CNN] AUC: %.4f[%.4f], AP: %.4f[%.4f], F1_score: %.4f[%.4f], F1_GradeIV: %.4f[%.4f]" % (np.array(rocauc_path_all).mean(),np.array(rocauc_path_all).std(), \
				np.array(ap_path_all).mean(),np.array(ap_path_all).std(), np.array(f1_micro_path_all).mean(),np.array(f1_micro_path_all).std(), np.array(f1_gradeIV_path_all).mean(),np.array(f1_gradeIV_path_all).std()))
			print("[Pathology CNN] F1_macro: %.4f[%.4f] , Recall: %.4f[%.4f], Precision: %.4f[%.4f], Kappa: %.4f[%.4f], MCC: %.4f[%.4f]" % (np.array(f1_macro_path_all).mean(),np.array(f1_macro_path_all).std(), \
				np.array(recall_macro_path_all).mean(),np.array(recall_macro_path_all).std(), np.array(precision_macro_path_all).mean(),np.array(precision_macro_path_all).std(), np.array(kappa_path_all).mean(), np.array(kappa_path_all).std(), np.array(mcc_path_all).mean(), np.array(mcc_path_all).std()))

			print("[Genomics SNN] AUC: %.4f[%.4f], AP: %.4f[%.4f], F1_score: %.4f[%.4f], F1_GradeIV: %.4f[%.4f]" % (np.array(rocauc_omic_all).mean(),np.array(rocauc_omic_all).std(), \
				np.array(ap_omic_all).mean(),np.array(ap_omic_all).std(), np.array(f1_micro_omic_all).mean(),np.array(f1_micro_omic_all).std(), np.array(f1_gradeIV_omic_all).mean(),np.array(f1_gradeIV_omic_all).std()))
			print("[Genomics SNN] F1_macro: %.4f[%.4f] , Recall: %.4f[%.4f], Precision: %.4f[%.4f], Kappa: %.4f[%.4f], MCC: %.4f[%.4f]" % (np.array(f1_macro_omic_all).mean(),np.array(f1_macro_omic_all).std(), \
				np.array(recall_macro_omic_all).mean(),np.array(recall_macro_omic_all).std(), np.array(precision_macro_omic_all).mean(),np.array(precision_macro_omic_all).std(), np.array(kappa_omic_all).mean(), np.array(kappa_omic_all).std(), np.array(mcc_omic_all).mean(), np.array(mcc_omic_all).std()))
	pickle.dump(results, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))