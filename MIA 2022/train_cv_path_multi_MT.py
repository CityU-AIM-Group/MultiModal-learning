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
from data_loaders_MT import pathomic_dataloader, pathomic_patches_dataloader
from options import parse_args
# from train_test_path_multi_MT import train, test
# from train_test_path_distill import train, test
from train_test_path_multi_distill_v2 import train, test

### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
# print("Key parameters:", "select pos pairs:", opt.select_pos_mode, "sample_KD:", opt.sample_KD, \
# 	"num_pos:", opt.nce_p2, "num_neg:", opt.nce_k, "CRD weight:", opt.CRD_weight)

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
results = []
rocauc_path_all, ap_path_all, f1_micro_path_all, f1_gradeIV_path_all = [], [], [], []
best_auc_all, best_ap_all, best_f1_all, best_f1_gradeIV_all = [], [], [], []
avg_auc_all, avg_ap_all, avg_f1_all, avg_f1_gradeIV_all = [], [], [], []

### 读取裁剪之后的每张ROI对应的9个patches.
roi_dir = 'all_st_patches_512'
data_cv_path_patches = '%s/splits_5cv_2022/gbmlgg5cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
print("Loading %s" % data_cv_path_patches)
data_cv_patches = pickle.load(open(data_cv_path_patches, 'rb'))
data_cv_splits_patches = data_cv_patches['cv_splits']

### 3. Sets-Up Main Loop
for k, data in data_cv_splits.items():
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
		fix_model, model, optimizer, metric_logger, best_test_metrics, avg_test_metrics = train(
			opt, train_loader, n_data, test_loader, test_loader_patches, device, k)

		### 3.2 Evalutes Train + Test Error, and Saves Model
		loss_test, cindex_test, pvalue_test, _, grad_acc_test, all_grad_metrics, pred_test, _, _ = test(
			opt, fix_model, model, test_loader_patches, device)

		if opt.task == 'surv':
			print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
			results.append(cindex_test)

		elif opt.task == 'grad':
			print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
			results.append(grad_acc_test)
			[rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path] = all_grad_metrics
			[best_auc, best_ap, best_f1, best_f1_gradeIV] = best_test_metrics
			[avg_auc, avg_ap, avg_f1, avg_f1_gradeIV] = avg_test_metrics

			rocauc_path_all.append(rocauc_path)
			ap_path_all.append(ap_path)
			f1_micro_path_all.append(f1_micro_path)
			f1_gradeIV_path_all.append(f1_gradeIV_path)

			best_auc_all.append(best_auc)
			best_ap_all.append(best_ap)
			best_f1_all.append(best_f1)
			best_f1_gradeIV_all.append(best_f1_gradeIV)

			avg_auc_all.append(avg_auc)
			avg_ap_all.append(avg_ap)
			avg_f1_all.append(avg_f1)
			avg_f1_gradeIV_all.append(avg_f1_gradeIV)

		### 3.3 Saves Model
		if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
			model_state_dict = model.module.cpu().state_dict()
		else:
			model_state_dict = model.cpu().state_dict()
		
		save_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k))
		torch.save({
			'split':k,
			'opt': opt,
			'epoch': opt.niter+opt.niter_decay,
			'data': data,
			'model_state_dict': model_state_dict,
			# 'model_state_dict': module_list.cpu().state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'metrics': metric_logger}, 
			save_path)

		print("saving the model at:", save_path)

		# pickle.dump(pred_train, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_train.pkl' % (opt.model_name, k, use_patch)), 'wb'))
		# pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch)), 'wb'))

		if opt.task == 'surv':
			print('Accuracy:', results)
			print("Average:", np.array(results).mean())

		elif opt.task == 'grad':
			# print("[Pathology CNN] AUC: %.4f, AP: %.4f, F1_score: %.4f, F1_GradeIV: %.4f" % (np.array(rocauc_path_all).mean(), \
			# 	np.array(ap_path_all).mean(), np.array(f1_micro_path_all).mean(), np.array(f1_gradeIV_path_all).mean()))
            
			print("[Pathology CNN Final] AUC: %.4f +/- %.4f, AP: %.4f +/- %.4f, F1_score: %.4f +/- %.4f, F1_GradeIV: %.4f +/- %.4f" % (
                np.array(rocauc_path_all).mean(), np.array(rocauc_path_all).std(), np.array(ap_path_all).mean(), \
                np.array(ap_path_all).std(), np.array(f1_micro_path_all).mean(), np.array(f1_micro_path_all).std(), \
                np.array(f1_gradeIV_path_all).mean(), np.array(f1_gradeIV_path_all).std())) 
			print("[Pathology CNN last3Avg] AUC: %.4f +/- %.4f, AP: %.4f +/- %.4f, F1_score: %.4f +/- %.4f, F1_GradeIV: %.4f +/- %.4f" % (
                np.array(avg_auc_all).mean(), np.array(avg_auc_all).std(), np.array(avg_ap_all).mean(), \
                np.array(avg_ap_all).std(), np.array(avg_f1_all).mean(), np.array(avg_f1_all).std(), \
                np.array(avg_f1_gradeIV_all).mean(), np.array(avg_f1_gradeIV_all).std())) 
			print("[Pathology CNN Best] AUC: %.4f +/- %.4f, AP: %.4f +/- %.4f, F1_score: %.4f +/- %.4f, F1_GradeIV: %.4f +/- %.4f" % (
                np.array(best_auc_all).mean(), np.array(best_auc_all).std(), np.array(best_ap_all).mean(), \
                np.array(best_ap_all).std(), np.array(best_f1_all).mean(), np.array(best_f1_all).std(), \
                np.array(best_f1_gradeIV_all).mean(), np.array(best_f1_gradeIV_all).std())) 

