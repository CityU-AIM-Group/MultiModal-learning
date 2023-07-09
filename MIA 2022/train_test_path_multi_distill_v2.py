"""
Author: Xing Xiaohan 
Date: 2022/12/17

从pathomic_fusion_20211126文件夹中拷贝的代码。复现MIA投稿中的结果。
"""

from functools import reduce
import random
import re
from torch.nn.modules import module
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import RandomSampler
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, recall_score, confusion_matrix

# from CL_utils.CRD_loss import CRDLoss
# from CL_utils.CRD_loss_v2 import CRDLoss
# from CL_utils.CRD_loss_v2 import CRDLoss_v2
from CL_utils.CRD_criterion_v3 import CRDLoss
from CL_utils.KD_losses import pred_KD_loss, SP_loss
from CL_utils.optimization import find_optimal_svm

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss, GNNLoss, feats_KL

from networks_new import define_net, define_reg, define_optimizer, define_scheduler
from utils import CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, count_parameters, sigmoid_rampup
from data_loaders_MT import omic_transform

#from GPUtil import showUtilization as gpu_usage
import pdb
import pickle
import os


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def GK_refine(optimizer, main_loss, loss_t_list, model_s):
    """
    2022/05/14
    之前是计算各种loss对于student featrues的梯度，然后根据这个梯度的相关性分配权重。
    现在改成计算各种loss对于student模型参数的梯度。
    """
    loss_t_list.append(main_loss)
    grads = []
    for obj in loss_t_list:
        # print("loss:", obj)
        optimizer.zero_grad(set_to_none=True)
        obj.backward(retain_graph=True)
        grad = []
        num_params = 0
        for name, param in model_s.named_parameters():
            # print(param.grad)
            if param.grad is not None:
                if 'fc_new2' not in name:
                    num_params += 1
                    grad.append(param.grad.clone())
        flatten_grad = torch.cat([g.flatten() for g in grad])
        # print("number of params:", num_params)
        # print(flatten_grad.shape)
        grads.append(flatten_grad)

    losses_div_tensor = torch.stack(loss_t_list) ##[5, 1]
    all_grads = torch.stack(grads).view(len(grads), -1) ##[5, d]
    grads_norm = torch.norm(all_grads, p=2, dim=1, keepdim=True)
    # print(grads_norm)
    grads_relation = torch.matmul(all_grads, all_grads.T)/torch.matmul(grads_norm, grads_norm.T)
    scale = torch.sum(grads_relation, dim=1)
    # print("gradient relation", grads_relation)
    # print("scale:", scale)

    total_KD_loss = torch.dot(scale.cuda()[:-1], losses_div_tensor.cuda()[:-1])

    return scale, total_KD_loss


def momentum_AEKD_loss(opt, optimizer, main_loss, feat_s, loss_t_list, mo_scale):
    """
    Date: 2022/05/23
    将KL div和CRD loss都对path分支的feat_s求梯度。path分支的CE loss也对feat_s求梯度。
    根据不同loss的梯度之间的一致性确定每个loss的权重。
    """
    grads = []
    feat_s.register_hook(lambda grad: grads.append(
        Variable(grad.data.clone(), requires_grad=False)))

    # print("KD loss:", loss_t_list)

    for loss_t in loss_t_list:
        optimizer.zero_grad()
        loss_t.backward(retain_graph=True)

    if opt.CE_grads:
        ### compute the gradients of cross entropy loss over the student feature.
        optimizer.zero_grad()
        main_loss.backward(retain_graph=True)

    losses_div_tensor = torch.stack(loss_t_list)
    all_grads = torch.stack(grads).view(len(grads), -1)
    grads_norm = torch.norm(all_grads, p=2, dim=1, keepdim=True)
    grads_relation = torch.matmul(all_grads, all_grads.T)/torch.matmul(grads_norm, grads_norm.T)
    if opt.grads_thresh == "True":
        grads_relation = torch.where(grads_relation>opt.thresh, 1.0, 0.0)
    scale = torch.sum(grads_relation, dim=1)
    # print("gradient relation", grads_relation)
    # print("scale:", scale)

    ### 将不同iteration的loss weights累积，得到mo_scale
    if mo_scale is None:
        mo_scale = scale
    else:
        mo_scale = opt.grads_m * mo_scale + (1-opt.grads_m)*scale

    if torch.cuda.is_available():
        mo_scale = mo_scale.cuda()
        losses_div_tensor.cuda()

    total_KD_loss = torch.dot(mo_scale[:-1], losses_div_tensor)

    return mo_scale, total_KD_loss


def AEKD_loss(opt, optimizer, main_loss, feat_s, loss_t_list):
    """
    将KL div和CRD loss都对path分支的feat_s求梯度。path分支的CE loss也对feat_s求梯度。
    根据one-class SVM算法对各种KD loss分配权重
    """
    grads = []
    feat_s.register_hook(lambda grad: grads.append(
        Variable(grad.data.clone(), requires_grad=False)))

    # print("KD loss:", loss_t_list)

    for loss_t in loss_t_list:
        optimizer.zero_grad()
        loss_t.backward(retain_graph=True)

    if opt.CE_grads:
        ### compute the gradients of cross entropy loss over the student feature.
        optimizer.zero_grad()
        main_loss.backward(retain_graph=True)


    # scale = find_optimal_svm(torch.stack(grads),
    #                         nu=0.5,
    #                         is_norm=opt.svm_norm)
    # losses_div_tensor = torch.stack(loss_t_list)
    # print("scale:", scale)
    # print("gradient:", torch.stack(grads).shape)


    losses_div_tensor = torch.stack(loss_t_list)
    all_grads = torch.stack(grads).view(len(grads), -1)
    grads_norm = torch.norm(all_grads, p=2, dim=1, keepdim=True)
    grads_relation = torch.matmul(all_grads, all_grads.T)/torch.matmul(grads_norm, grads_norm.T)
    scale = torch.sum(grads_relation, dim=1)
    # print("gradient relation", grads_relation)
    # print("scale:", scale)

    if torch.cuda.is_available():
        scale = scale.cuda()
        losses_div_tensor.cuda()

    # if scale[-1] > 0.3:
    #     return torch.dot(scale[:len(logit_t_list)], losses_div_tensor)
    # else:
    #     return 0.0

    total_KD_loss = torch.dot(scale[:-1], losses_div_tensor)

    return scale, total_KD_loss



def AEKD_loss_v2(opt, optimizer, main_loss, feat_s, KD_loss_list):
    """
    将KL div和CRD loss都对path分支的feat_s求梯度。path分支的CE loss也对feat_s求梯度。
    计算各种KD loss的梯度和CE loss的gradient similarity，
    去除similarity小于0的梯度，如果和CE loss的gradient similarity大于0，则保存该loss。
    """
    grads = []
    feat_s.register_hook(lambda grad: grads.append(
        Variable(grad.data.clone(), requires_grad=False)))

    if opt.sample_KD == "True":
        loss_t_list = [torch.sum(KD_loss_list[i])/opt.batch_size for i in range(len(KD_loss_list))]
    else:
        loss_t_list = KD_loss_list

    # print("KD loss:", loss_t_list)

    for loss_t in loss_t_list:
        optimizer.zero_grad()
        loss_t.backward(retain_graph=True)

    if opt.CE_grads:
        ### compute the gradients of cross entropy loss over the student feature.
        optimizer.zero_grad()
        main_loss.backward(retain_graph=True)


    if opt.sample_KD == "True":
        """
        对于batch中每个样本，计算每种KD loss的梯度和CE loss梯度的相似度。
        过滤掉相似度小于0的KD loss, 将剩余的相似度作为每个样本的各种KD loss的权重。
        得到权重 scale: [batch_size, num_KD_losses].
        """
        grads_all = torch.permute(torch.stack(grads), (1,0,2))
        grads_norm = torch.norm(grads_all, p=2, dim=2, keepdim=True)
        KD_loss_grads, CE_loss_grads = grads_all[:,:-1,:], torch.unsqueeze(grads_all[:,-1,:], 2)
        KD_loss_grads_norm, CE_loss_grads_norm = grads_norm[:,:-1,:], torch.unsqueeze(grads_norm[:,-1,:], 2)
    
    else:
        """
        对于batch中所有样本，计算每种KD loss的梯度和CE loss梯度的相似度。
        过滤掉相似度小于0的KD loss, 将剩余的相似度作为这个batch中各种KD loss的权重。
        得到权重 scale: [num_KD_losses].
        """
        grads_all = torch.reshape(torch.stack(grads), (len(grads), -1)) # [5, bs*feat_dim]
        grads_norm = torch.norm(grads_all, p=2, dim=1, keepdim=True)
        KD_loss_grads, CE_loss_grads = grads_all[:-1], torch.unsqueeze(grads_all[-1], 1)
        KD_loss_grads_norm, CE_loss_grads_norm = grads_norm[:-1], torch.unsqueeze(grads_norm[-1], 1)

    grads_similarity = torch.matmul(KD_loss_grads, CE_loss_grads)
    grads_similarity = torch.squeeze(grads_similarity/torch.matmul(KD_loss_grads_norm, CE_loss_grads_norm))
    # scale = torch.relu(grads_similarity) * len(loss_t_list) # (bs, 4)
    # scale = torch.where(grads_similarity > 0, 1.0/len(KD_loss_list), 0.0)
    scale = torch.where(grads_similarity > 0, 1.0, 0.0)
    # scale = torch.where(grads_similarity > 0, grads_similarity, 0.0)
    # print("gradient similarity:", grads_similarity)

    losses_div_tensor = torch.stack(KD_loss_list).T

    if torch.cuda.is_available():
        scale = scale.cuda()
        losses_div_tensor.cuda()

    if opt.sample_KD == "True":
        total_KD_loss = torch.sum(torch.multiply(scale, losses_div_tensor))/opt.batch_size
    else:
        total_KD_loss = torch.sum(torch.multiply(scale, losses_div_tensor))
    
    # print("loss weights:", scale)
    # print("KD losses:", losses_div_tensor, total_KD_loss)

    return total_KD_loss


def train(opt, train_loader, n_data, test_loader, test_loader_patches, device, k):
    cudnn.deterministic = True
    seed = 2019
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    def get_current_consistency_weight(epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return sigmoid_rampup(epoch, opt.consistency_rampup)


    ### load pretrained model parameters.
    load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.fixed_model, '%s_%d_best.pt' % (opt.fixed_model, k))
    model_ckpt = torch.load(load_path, map_location=device)
    model_state_dict = model_ckpt['model_state_dict']
    if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata

    fix_model = define_net(opt, k)
    if isinstance(fix_model, torch.nn.DataParallel): fix_model = fix_model.module

    print('Loading the model from %s' % load_path)
    fix_model.load_state_dict(model_state_dict)

    fix_model = torch.nn.DataParallel(fix_model).to(device)

    for param in fix_model.parameters():
        # print("fix model parameters")
        param.detach_()
        param.requires_grad = False


    def create_model(ema=False):
        model = define_net(opt, k, path_only=True).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    print("create path model")
    model = create_model()
    print("create path EMA model")
    ema_model = create_model(ema=True)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        ema_model = torch.nn.DataParallel(ema_model)
    model = model.to(device)
    ema_model = ema_model.to(device)

    module_list = nn.ModuleList([])
    module_list.append(model)


    """不同类型的knowledge distillation"""
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'feats_KL':
        criterion_kd = feats_KL()        
    elif opt.distill == 'crd':
        criterion_kd = CRDLoss(opt, n_data).to(device)
        # opt.n_data = n_data
        # criterion_kd = CRDLoss(opt).to(device)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)

        criterion_kd_path = CRDLoss(opt, n_data).to(device)
        module_list.append(criterion_kd_path.embed_s)
        module_list.append(criterion_kd_path.embed_t)

        # ### 单向DC-Distill
        # criterion_kd = CRDLoss_v2(opt, n_data).to(device)
        # criterion_kd_path = CRDLoss_v2(opt, n_data).to(device)
        # module_list.append(criterion_kd.embed_s)
        # module_list.append(criterion_kd_path.embed_s)

    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()        
    elif opt.distill == 'hkd':
        opt.n_data = n_data
        criterion_kd = GNNLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        module_list.append(criterion_kd.gnn_s)
        module_list.append(criterion_kd.gnn_t)


    # optimizer = define_optimizer(opt, model)
    optimizer = define_optimizer(opt, module_list)
    scheduler = define_scheduler(opt, optimizer)
    print(module_list)
    print("Number of Trainable Parameters: %d" % count_parameters(module_list))
    print("Activation Type:", opt.act_type)
    print("Optimizer Type:", opt.optimizer_type)
    print("Regularization Type:", opt.reg_type)

    use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_vgg_features else ('_', 'all_st')
    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[]}}
    
    iter_num = opt.global_step

    best_acc = 0.0
    avg_all_metrics = np.array([0.0, 0.0, 0.0, 0.0])
    scale = None

    for epoch in tqdm(range(opt.epoch_count, opt.niter+opt.niter_decay+1)):

        print("learning rate:", scheduler.get_lr())
        module_list.train()
        fix_model.train()
        # ema_model.train()
        risk_pred_all, risk_path_all, risk_omic_all = np.array([]), np.array([]), np.array([])  # Used for calculating the C-Index
        censor_all, survtime_all = np.array([]), np.array([])
        loss_epoch, loss_fuse_epoch, loss_path_epoch, loss_omic_epoch = 0, 0, 0, 0
        grad_acc_epoch, grad_path_epoch, grad_omic_epoch = 0, 0, 0
        loss_cls_epoch, loss_div_epoch, loss_kd_epoch = 0, 0, 0
        loss_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        # loss_weights = np.array([0.0, 0.0, 0.0])

        for batch_idx, ((x_path, ema_x_path), x_grph, x_omic, censor, survtime, \
            grade, index, sample_idx) in enumerate(train_loader):
            
            censor = censor.to(device) if "surv" in opt.task else censor
            grade = grade.to(device) if "grad" in opt.task else grade

            # print("batch index:", index)
            # print("path input:", x_path.shape)
            # print("input disturbance:", x_path - ema_x_path)

            ### student model
            _, path_feat, logit_path, pred_path, _ = model(
                x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))
            # print("logit:", logit_path)
            # print("predictions:", pred_path)
            ### self mean teacher model
            with torch.no_grad():
                _, ema_path_feat, ema_logit_path, ema_pred_path, _ = ema_model(
                    x_path=ema_x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))
                fuse_feat, _, _, _, logits, pred, _, _, _, _, _ = fix_model(
                    x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))

            # print("fuse features:", torch.max(fuse_feat), torch.mean(fuse_feat))
            # print("path features:", torch.max(path_feat), torch.mean(path_feat))
            # print("path logits:", logit_path)
            # print("fuse logits:", logits[-1])
            # print("fuse pred:", pred)

            loss_cox = CoxLoss(survtime, censor, pred_path, device) if opt.task == "surv" else 0

            ### for the grading task, xiaohan added in 2021.12.22
            loss_cls = F.nll_loss(pred_path, grade) if opt.task == "grad" else 0
            
            if opt.num_teachers == 2:
                loss_div1 = criterion_div(logit_path, logits[-1].detach())
                loss_div2 = criterion_div(logit_path, ema_logit_path.detach())
                loss_div = loss_div1 + loss_div2
            elif opt.num_teachers == 1 and opt.which_teacher == "fuse":
                loss_div = criterion_div(logit_path, logits[-1].detach())
            elif opt.num_teachers == 1 and opt.which_teacher == "self_EMA":
                loss_div = criterion_div(logit_path, ema_logit_path.detach())

            # other kd beyond KL divergence
            if opt.distill == 'kd':
                loss_kd = 0
            elif opt.distill == 'feats_KL':
                loss_kd = criterion_kd(path_feat, fuse_feat.detach())
            elif opt.distill == 'crd':
                # print("contrast index:", sample_idx.shape)
                if opt.num_teachers == 2:
                    loss_kd1 = criterion_kd(epoch/opt.niter_decay, path_feat, fuse_feat.detach(), index.cuda(), sample_idx.cuda())
                    loss_kd2 = criterion_kd_path(epoch/opt.niter_decay, path_feat, ema_path_feat.detach(), index.cuda(), sample_idx.cuda())
                    loss_kd = loss_kd1 + loss_kd2
                elif opt.num_teachers == 1 and opt.which_teacher == "fuse":
                    loss_kd = criterion_kd(epoch/opt.niter_decay, path_feat, fuse_feat.detach(), index.cuda(), sample_idx.cuda())
                elif opt.num_teachers == 1 and opt.which_teacher == "self_EMA":
                    loss_kd = criterion_kd(epoch/opt.niter_decay, path_feat, ema_path_feat.detach(), index.cuda(), sample_idx.cuda())

            elif opt.distill == 'rkd':
                loss_kd = criterion_kd(path_feat, fuse_feat.detach())
            elif opt.distill == 'pkt':
                loss_kd = criterion_kd(path_feat, fuse_feat.detach())
            elif opt.distill == 'similarity':
                loss_kd = criterion_kd(path_feat, fuse_feat.detach())
            elif opt.distill == 'hkd':
                loss_kd = criterion_kd(epoch, path_feat, logit_path, fuse_feat.detach(), \
                    logits[-1].detach(), index.cuda(), sample_idx.cuda())
            else:
                raise NotImplementedError(opt.distill)

            """
            根据各种loss的梯度一致性给不同的知识分配权重。
            """
            if opt.num_teachers == 2:
                loss_div1 = opt.alpha * loss_div1
                loss_div2 = opt.alpha * loss_div2
                if opt.distill == 'crd':
                    loss_kd1 = opt.beta * loss_kd1
                    loss_kd2 = opt.beta * loss_kd2
                    KD_loss_list = [loss_div1, loss_div2, loss_kd1, loss_kd2]
                elif opt.distill == 'kd':
                    KD_loss_list = [loss_div1, loss_div2]

            if opt.assign_weights == "True":
                # loss_KD = AEKD_loss_v2(opt, optimizer, loss_cls, path_feat, KD_loss_list) * len(KD_loss_list)
                # scale, loss_KD = GK_refine(optimizer, loss_cls, KD_loss_list, module_list[0])

                # scale, loss_KD = AEKD_loss(opt, optimizer, loss_cls, path_feat, KD_loss_list)
                scale, loss_KD = momentum_AEKD_loss(opt, optimizer, loss_cls, path_feat, KD_loss_list, scale)
                loss_weights += scale.detach().cpu().numpy()
                if opt.grads_thresh == "False":
                    loss_KD = loss_KD * len(KD_loss_list)
                # print("loss weight for current batch:", scale.detach().cpu().numpy())
                # print("distillation losses:", loss_div, loss_kd, loss_KD)
            else:
                loss_KD = opt.alpha * loss_div + opt.beta * loss_kd


            loss_reg = define_reg(opt, model)
            loss = opt.lambda_cox * loss_cox + opt.lambda_nll * loss_cls + opt.lambda_reg * loss_reg + loss_KD

            loss_kd = opt.beta * loss_kd

            loss_epoch += loss.data.item()
            loss_cls_epoch += loss_cls.item()

            if loss_div > 0.0:
                loss_div_epoch += loss_div.item()

            if loss_kd > 0.0:
                loss_kd_epoch += loss_kd.item()

            optimizer.zero_grad()
            # ### newly added.
            # path_feat.retain_grad()
            # loss_kd.backward(retain_graph=True)
            # print(path_feat.grad)
            loss.backward()
            # print(path_feat.grad)
            optimizer.step()
            update_ema_variables(model, ema_model, opt.ema_decay, iter_num)
            iter_num = iter_num + 1
            
            if opt.task == "surv":
                risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
                risk_path_all = np.concatenate((risk_path_all, pred_path.detach().cpu().numpy().reshape(-1)))
                censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
                survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information
            
            elif opt.task == "grad":
                pred_path = pred_path.argmax(dim=1, keepdim=True)
                grad_path_epoch += pred_path.eq(grade.view_as(pred_path)).sum().item()

            if opt.verbose > 0 and opt.print_every > 0 and (batch_idx % opt.print_every == 0 or batch_idx+1 == len(train_loader)):
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch+1, opt.niter+opt.niter_decay, batch_idx+1, len(train_loader), loss.item()))

        scheduler.step()
        print("loss weights:", loss_weights / len(train_loader))

        if opt.measure or epoch == (opt.niter+opt.niter_decay - 1):
            loss_cls_epoch /= len(train_loader)
            loss_div_epoch /= len(train_loader)
            loss_kd_epoch /= len(train_loader)
            

            cindex_epoch = CIndex_lifeline(risk_path_all, censor_all, survtime_all) if opt.task == 'surv' else None

            ### histological grading.
            grad_acc_epoch = grad_path_epoch / len(train_loader.dataset) if opt.task == 'grad' else None

            ### test the model.
            if epoch > (opt.niter+opt.niter_decay - 20):
                ### 最后三个周期用全部的patches进行测试，每个病理图像对应9个patches.
                loss_test, cindex_test, _, _, grad_acc_test, all_grad_metrics, pred_test, _, _ = test(
                        opt, fix_model, model, test_loader_patches, device)
            else:
                loss_test, cindex_test, _, _, grad_acc_test, all_grad_metrics, pred_test, _, _ = test(
                        opt, fix_model, model, test_loader, device)

            if epoch > opt.niter_decay-3:
                avg_all_metrics += np.array(all_grad_metrics)

            # pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%s%d_pred_test.pkl' % (opt.model_name, k, use_patch, epoch)), 'wb'))

            if opt.verbose > 0:
                if opt.task == 'surv':
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test))
                elif opt.task == 'grad':
                    print('[{:s}]\t\t KL div Loss: {:.4f}'.format('Train', loss_div_epoch))
                    print('[{:s}]\t\t KD Loss: {:.4f}'.format('Train', loss_kd_epoch))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_cls_epoch, 'Accuracy', grad_acc_epoch))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'Accuracy', grad_acc_test))

            if opt.task == 'grad' and loss_epoch < opt.patience:
                print("Early stopping at Epoch %d" % epoch)
                break

            if epoch > opt.niter_decay-20:
                avg_metric = np.mean(np.array(all_grad_metrics))
                if avg_metric > best_acc:
                    best_metrics = all_grad_metrics
                    best_acc = avg_metric
                    save_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, \
                        '%s_%d_best.pt' % (opt.model_name, k))
                    model_state_dict = model.module.state_dict()
                    torch.save({
                        'split':k,
                        'opt': opt,
                        'epoch': opt.niter+opt.niter_decay,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': metric_logger}, 
                        save_path)
                    print("saving the model at:", save_path)

    avg_metrics = avg_all_metrics/3.0
    print("average metrics [pathology CNN]:", avg_all_metrics/3.0)   

    return fix_model, model, optimizer, metric_logger, best_metrics, avg_metrics


def test(opt, fix_model, model, test_loader, device):
    fix_model.eval()
    model.eval()
    risk_pred_all, risk_path_all, risk_omic_all = np.array([]), np.array([]), np.array([])  # Used for calculating the C-Index
    censor_all, survtime_all = np.array([]), np.array([])

    probs_all, probs_path, probs_omic, gt_all = None, None, None, np.array([])
    loss_test, loss_fuse_test, loss_path_test, loss_omic_test = 0, 0, 0, 0
    grad_fuse_test, grad_path_test, grad_omic_test = 0, 0, 0
    grads_fuse_all, grads_path_all, grads_omic_all = None, None, None
    feat_fuse_all, feat_path_all, feat_omic_all = None, None, None

    for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade) in enumerate(test_loader):

        censor = censor.to(device) if "surv" in opt.task else censor
        grade = grade.to(device) if "grad" in opt.task else grade

        _, feat_path, _, pred_path, grads_path = model(
            x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))

        with torch.no_grad():
            fuse_feat, _, _, _, _, pred, _, _, _, _, _ = fix_model(
                x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))

        loss_cox = CoxLoss(survtime, censor, pred_path, device) if opt.task == "surv" else 0

        loss_reg = define_reg(opt, model)

        ### for the grading task, xiaohan added in 2021.12.09
        loss_nll = F.nll_loss(pred_path, grade) if opt.task == "grad" else 0

        loss = opt.lambda_cox*loss_cox + opt.lambda_nll*loss_nll + opt.lambda_reg*loss_reg
        loss_test += loss.data.item()

        gt_all = np.concatenate((gt_all, grade.detach().cpu().numpy().reshape(-1)))   # Logging Information

        feat_path = feat_path.detach().cpu().numpy()
        feat_path_all = feat_path if feat_path_all is None else np.concatenate((feat_path_all, feat_path), axis=0)

        if opt.return_grad == "True":
            grads_path_all = grads_path if grads_path_all is None else np.concatenate((grads_path_all, grads_path), axis=0)

        if opt.task == "surv":
            risk_path_all = np.concatenate((risk_path_all, pred_path.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information
        
        elif opt.task == "grad":
            grad_path_test, probs_path = compute_accuracy(pred_path, grade, probs_path, grad_path_test)
            grad_fuse_test, probs_all = compute_accuracy(pred, grade, probs_all, grad_fuse_test)


    ################################################### 
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader)
    loss_fuse_test /= len(test_loader)
    loss_path_test /= len(test_loader)
    loss_omic_test /= len(test_loader)
    cindex_path = CIndex_lifeline(risk_path_all, censor_all, survtime_all) if opt.task == 'surv' else None
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all) if opt.task == 'surv' else None

    grad_path_test = grad_path_test / len(test_loader.dataset) if opt.task == 'grad' else None

    """compute other metrics for the grading task"""
    if opt.task == "grad":
        enc = LabelBinarizer()
        enc.fit(gt_all)
        grad_gt = enc.transform(gt_all)
        # print(grad_gt.shape, probs_all.shape)

        rocauc_fuse, ap_fuse, f1_micro_fuse, f1_gradeIV_fuse = grading_metrics(grad_gt, probs_all)
        rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path = grading_metrics(grad_gt, probs_path)
        print("fixed fuse branch:", rocauc_fuse, ap_fuse, f1_micro_fuse, f1_gradeIV_fuse)
        print("Path branch:", rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path)

        all_grad_metrics = [rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path]
    else:
        all_grad_metrics = None

    ### save the predictions from the three branches and the ground truth in this list.
    pred_test = [risk_pred_all, risk_path_all, risk_omic_all, survtime_all, censor_all, \
        probs_all, probs_path, probs_omic, gt_all]
    grads_test = [grads_fuse_all, grads_path_all, grads_omic_all]
    feats_test = [feat_fuse_all, feat_path_all, feat_omic_all, gt_all]

    # return loss_test, loss_fuse_test, loss_path_test, loss_omic_test, cindex_test, cindex_path, cindex_omic, pvalue_test, \
    #     surv_acc_test, grad_acc_test, grad_path_test, grad_omic_test, all_grad_metrics, pred_test, grads_test, feats_test

    return loss_test, cindex_path, pvalue_test, surv_acc_test, grad_path_test, \
         all_grad_metrics, pred_test, grads_test, feats_test



def compute_accuracy(preds, labels, probs_all, grad_acc_test):
    """
    Compute the grading accuracy and return the predicted probs.
    """
    grade_pred = preds.argmax(dim=1, keepdim=True)
    grad_acc_test += grade_pred.eq(labels.view_as(grade_pred)).sum().item()
    probs_np = preds.detach().cpu().numpy()
    probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)

    return grad_acc_test, probs_all


def grading_metrics(y_label, y_pred, avg='micro'):
    rocauc = roc_auc_score(y_label, y_pred, average = avg)
    ap = average_precision_score(y_label, y_pred, average=avg)

    f1_micro = f1_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1), average=avg)
    f1_gradeIV = f1_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1), average=None)[2]
    print("f1-score:", f1_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1), average=None))

    # print("confusion matrix:", confusion_matrix(np.argmax(y_label, axis=1), y_pred.argmax(axis=1)))

    return rocauc, ap, f1_micro, f1_gradeIV



def test_model(opt, model, test_loader, device):
    model.eval()
    risk_pred_all, risk_path_all, risk_omic_all = np.array([]), np.array([]), np.array([])  # Used for calculating the C-Index
    censor_all, survtime_all = np.array([]), np.array([])

    probs_all, probs_path, probs_omic, gt_all = None, None, None, np.array([])
    loss_test, loss_fuse_test, loss_path_test, loss_omic_test = 0, 0, 0, 0
    grad_fuse_test, grad_path_test, grad_omic_test = 0, 0, 0
    grads_fuse_all, grads_path_all, grads_omic_all = None, None, None
    feat_fuse_all, feat_path_all, feat_omic_all = None, None, None

    for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade) in enumerate(test_loader):

        censor = censor.to(device) if "surv" in opt.task else censor
        grade = grade.to(device) if "grad" in opt.task else grade

        _, feat_path, logit_path, pred_path, grads_path = model(
            x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))

        # print("path test feature:", feat_path.shape, torch.count_nonzero(feat_path))
        # print("path features:", torch.max(feat_path), torch.min(feat_path), torch.mean(feat_path))

        loss_cox = CoxLoss(survtime, censor, pred_path, device) if opt.task == "surv" else 0

        loss_reg = define_reg(opt, model)

        ### for the grading task, xiaohan added in 2021.12.09
        loss_nll = F.nll_loss(pred_path, grade) if opt.task == "grad" else 0

        loss = opt.lambda_cox*loss_cox + opt.lambda_nll*loss_nll + opt.lambda_reg*loss_reg
        loss_test += loss.data.item()

        gt_all = np.concatenate((gt_all, grade.detach().cpu().numpy().reshape(-1)))   # Logging Information

        feat_path = feat_path.detach().cpu().numpy()
        feat_path_all = feat_path if feat_path_all is None else np.concatenate((feat_path_all, feat_path), axis=0)

        if opt.return_grad == "True":
            grads_path_all = grads_path if grads_path_all is None else np.concatenate((grads_path_all, grads_path), axis=0)

        if opt.task == "surv":
            risk_path_all = np.concatenate((risk_path_all, pred_path.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information
        
        elif opt.task == "grad":
            grad_path_test, probs_path = compute_accuracy(pred_path, grade, probs_path, grad_path_test)

    ################################################### 
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader)
    loss_fuse_test /= len(test_loader)
    loss_path_test /= len(test_loader)
    loss_omic_test /= len(test_loader)
    cindex_path = CIndex_lifeline(risk_path_all, censor_all, survtime_all) if opt.task == 'surv' else None
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all) if opt.task == 'surv' else None

    grad_path_test = grad_path_test / len(test_loader.dataset) if opt.task == 'grad' else None

    """compute other metrics for the grading task"""
    if opt.task == "grad":
        enc = LabelBinarizer()
        enc.fit(gt_all)
        grad_gt = enc.transform(gt_all)
        # print(grad_gt.shape, probs_all.shape)

        rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path = grading_metrics(grad_gt, probs_path)
        print("Path branch:", rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path)
        all_grad_metrics = [rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path]
    else:
        all_grad_metrics = None

    ### save the predictions from the three branches and the ground truth in this list.
    pred_test = [risk_pred_all, risk_path_all, risk_omic_all, survtime_all, censor_all, \
        probs_all, probs_path, probs_omic, gt_all]
    grads_test = [grads_fuse_all, grads_path_all, grads_omic_all]
    feats_test = [feat_fuse_all, feat_path_all, feat_omic_all, gt_all]

    return loss_test, cindex_path, pvalue_test, surv_acc_test, grad_path_test, \
         all_grad_metrics, pred_test, grads_test, feats_test
