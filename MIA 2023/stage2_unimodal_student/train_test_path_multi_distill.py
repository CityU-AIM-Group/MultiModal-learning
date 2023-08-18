"""
Author: Xing Xiaohan 
Date: 2023/08/13
对比不同的multi-task learning方法给多个loss分配权重的效果。
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
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.metrics.pairwise import cosine_similarity

from CL_utils.CRD_criterion_v10 import CRDLoss


from KD_loss import DistillKL, SP_loss
from networks_new import define_net, define_reg, define_optimizer, define_scheduler
from utils import CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, count_parameters, sigmoid_rampup


#from GPUtil import showUtilization as gpu_usage
import pdb
import pickle
import os


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



def GK_refine(opt, optimizer, main_loss, feat_s, loss_t_list):
    """
    Compute the gradient of each loss and assign weights according to their consistency.
    """
    grads = []
    feat_s.register_hook(lambda grad: grads.append(
        Variable(grad.data.clone(), requires_grad=False)))

    for loss_t in loss_t_list:
        optimizer.zero_grad()
        loss_t.backward(retain_graph=True)

    if opt.CE_grads:
        ### compute the gradients of cross entropy loss over the student feature.
        optimizer.zero_grad()
        main_loss.backward(retain_graph=True)

    # print("loss_t_list:", loss_t_list)
    losses_div_tensor = torch.stack(loss_t_list)
    all_grads = torch.stack(grads).view(len(grads), -1)
    grads_norm = torch.norm(all_grads, p=2, dim=1, keepdim=True)
    # grads_relation = torch.matmul(all_grads, all_grads.T)*len(loss_t_list)/torch.matmul(grads_norm, grads_norm.T)
    grads_relation = torch.matmul(all_grads, all_grads.T)/torch.matmul(grads_norm, grads_norm.T)
    scale = torch.sum(grads_relation, dim=1)

    if torch.cuda.is_available():
        scale = scale.cuda()
        losses_div_tensor.cuda()
        
    total_KD_loss = torch.dot(scale[:-1], losses_div_tensor)

    return scale, total_KD_loss



def GK_refine_thresh(opt, optimizer, main_loss, feat_s, loss_t_list):
    """
    Date: 2023/01/30 
    将KL div和CRD loss都对path分支的feat_s求梯度。path分支的CE loss也对feat_s求梯度。
    根据不同loss的梯度之间的一致性确定每个loss的权重。
    如果一致性大于thresh, 则设置为1. 否则设置为0.
    之前都是对每个batch分配不同teacher的权重, 现在改成每个样本都对不同teacher分配权重。
    """
    grads = []
    feat_s.register_hook(lambda grad: grads.append(
        Variable(grad.data.clone(), requires_grad=False)))

    for loss_t in loss_t_list:
        # print("loss_t:", loss_t)
        optimizer.zero_grad()
        loss_t.sum().backward(retain_graph=True)

    if opt.CE_grads:
        ### compute the gradients of cross entropy loss over the student feature.
        optimizer.zero_grad()
        main_loss.backward(retain_graph=True)

    losses_div_tensor = torch.stack(loss_t_list)

    all_grads = torch.stack(grads)
    all_scale = torch.zeros(opt.batch_size, len(grads))
    for i in range(opt.batch_size):
        sample_grad_relation = cosine_similarity(all_grads[:, i, :].cpu().detach().numpy())
        # sample_grad_relation = np.sum(np.where(sample_grad_relation>opt.grads_thresh, 1.0, 0.0), 0)
        # sample_grad_relation = np.sum(sample_grad_relation, 0)
        if opt.use_grads_thresh == 'False':
            # print("without thresholding the gradient relation.")
            sample_grad_relation = np.sum(np.where(sample_grad_relation>0, sample_grad_relation, 0.0), 0)
        elif opt.use_grads_thresh == 'True':
            # print("using grads_thresh to binarize the gradient relation")
            sample_grad_relation = np.sum(np.where(sample_grad_relation>opt.grads_thresh, 1.0, 0.0), 0)
        all_scale[i] = torch.tensor(sample_grad_relation)

    if torch.cuda.is_available():
        all_scale = all_scale.cuda()
        losses_div_tensor.cuda()
    
    total_KD_loss = torch.sum(torch.multiply(all_scale[:, :-1].transpose(0, 1), losses_div_tensor))/opt.batch_size
    scale = all_scale.mean(0)
    # print("loss scale:", scale)
    # print("total_KD_loss:", total_KD_loss)

    return scale, total_KD_loss


def assign_sample_weights(pred_s, pred_t, gt, discrep_scale, max_discrep):
    """
    xxh, 2022/12/27
    Compute the probabilistic margin (confidence) in the teacher and student,
    assign weights to query samples according to their discrepancy in the teacher and student model.
    pred_s, pred_t, gt: (batch_size, num_class)
    """
    gt = F.one_hot(gt, 3).float()
    # print("pred_t:", pred_t)
    gt_prob_t = torch.sum(pred_t*gt, 1)
    top2_prob_t = torch.max(pred_t*(1-gt), 1)[0]
    # conf_t = gt_prob_t - top2_prob_t
    conf_t = torch.log(gt_prob_t) - torch.log(top2_prob_t)

    gt_prob_s = torch.sum(pred_s*gt, 1)
    top2_prob_s = torch.max(pred_s*(1-gt), 1)[0]
    # conf_s = gt_prob_s - top2_prob_s
    conf_s = torch.log(gt_prob_s) - torch.log(top2_prob_s)

    # print("conf_t and conf_s:", conf_t, conf_s)
    discrepancy = torch.maximum(conf_t - conf_s, torch.zeros_like(conf_t)).detach()
    # print("original query discrepancy:", discrepancy)
    discrepancy = torch.minimum(discrepancy, max_discrep*torch.ones_like(conf_t)) ## 设置最大discrepancy值对query reweight系数进行截断。
    # print("max discrepancy:", discrepancy.max())
    # discrepancy = torch.minimum(discrep_scale * discrepancy, torch.ones_like(conf_t))
    # print("query discrepancy:", discrepancy)

    return discrepancy



def intra_inter_similarity(feature_similarity, intra_inter_indicater):
    """
    intra_inter_indicater中1表示两个同类别样本, 0表示两个不同类别样本。
    """
    intra_similarity = feature_similarity[intra_inter_indicater == 1].mean()
    inter_similarity = feature_similarity[intra_inter_indicater == 0].mean()

    return intra_similarity, inter_similarity


def evaluate_feature(fuse_features, path_features, labels):
    fuse_similarity = cosine_similarity(fuse_features)
    path_similarity = cosine_similarity(path_features)
    num_data = labels.shape[0]
    intra_inter_indicater = 1.0 * np.equal(np.tile(labels.reshape(-1, 1), (1, num_data)), \
        np.tile(labels.reshape(1, -1), (num_data, 1)))

    intra_similarity, inter_similarity = intra_inter_similarity(fuse_similarity, intra_inter_indicater)
    print("\n[Teacher] intra-class similarity:", intra_similarity)
    print("[Teacher] inter-class similarity:", inter_similarity)

    intra_similarity, inter_similarity = intra_inter_similarity(path_similarity, intra_inter_indicater)
    print("[Student] intra-class similarity:", intra_similarity)
    print("[Student] inter-class similarity:", inter_similarity)

    similarity_diff = np.mean(np.abs(fuse_similarity - path_similarity))
    print("Difference between the similarity matrices of the teacher and student features:", similarity_diff)


def evaluate_logits(fuse_preds, path_preds, train_class_idx):
    fuse_similarity = cosine_similarity(fuse_preds)
    path_similarity = cosine_similarity(path_preds)

    similarity_diff = np.mean(np.abs(fuse_similarity - path_similarity))
    print("Difference between the similarity matrices of the teacher and student logits:", similarity_diff)


def train(opt, train_loader, train_class_idx, n_data, test_loader, test_loader_patches, device, k):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2019)
    torch.manual_seed(2019)
    random.seed(2019)
    np.random.seed(2019)
    

    ### load pretrained model parameters.
    # load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.fixed_model, '%s_%d_best.pt' % (opt.fixed_model, k))
    load_path = os.path.join(opt.checkpoints_dir, opt.fixed_model, 'stage1_pathomic_teacher_%d_best.pt' % (k))

    model_ckpt = torch.load(load_path, map_location=device)
    model_state_dict = model_ckpt['model_state_dict']
    if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata

    fix_model = define_net(opt, k)
    if isinstance(fix_model, torch.nn.DataParallel): fix_model = fix_model.module

    print('Loading the model from %s' % load_path)
    fix_model.load_state_dict(model_state_dict)

    fix_model = torch.nn.DataParallel(fix_model).to(device)
    # print("fixed multi-modal network:", fix_model)

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

    model = create_model()
    ema_model = create_model(ema=True)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        ema_model = torch.nn.DataParallel(ema_model)
    model = model.to(device)
    ema_model = ema_model.to(device)

    module_list = nn.ModuleList([])
    module_list.append(model)


    # """不同类型的knowledge distillation"""
    criterion_div = DistillKL(opt.kd_T)

    criterion_sp = SP_loss()
    # if opt.distill == 'kd':
    #     criterion_kd = DistillKL(opt.kd_T)
    # elif opt.distill == 'crd':

    criterion_kd = CRDLoss(opt, n_data, train_class_idx).to(device)
    module_list.append(criterion_kd.embed_s)
    module_list.append(criterion_kd.embed_t)

    criterion_kd_path = CRDLoss(opt, n_data, train_class_idx).to(device)
    module_list.append(criterion_kd_path.embed_s)
    module_list.append(criterion_kd_path.embed_t)

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
    avg_all_metrics = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    teacher1_all_sample_weights = []
    teacher2_all_sample_weights = []

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
        # loss_weights = np.array([0.0, 0.0, 0.0, 0.0])

        fuse_features_all = np.zeros((n_data, opt.mmhid))
        path_features_all = np.zeros((n_data, opt.path_dim))

        fuse_preds_all = np.zeros((n_data, opt.label_dim))
        path_preds_all = np.zeros((n_data, opt.label_dim))


        for batch_idx, ((x_path, ema_x_path), x_grph, x_omic, censor, survtime, \
            grade, index, sample_idx) in enumerate(train_loader):

            # print("x_omic:", x_omic.shape)
            
            censor = censor.to(device) if "surv" in opt.task else censor
            grade = grade.to(device) if "grad" in opt.task else grade

            # print("query index:", index)
            # print("contrastive samples index:", sample_idx)

            ### student model
            _, path_feat, logit_path, pred_path, _ = model(
                x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))

            ### self mean teacher model
            with torch.no_grad():
                _, ema_path_feat, ema_logit_path, ema_pred_path, _ = ema_model(
                    x_path=ema_x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))
                # _, ema_path_feat, ema_logit_path, ema_pred_path, _ = ema_model(
                #     x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))
                fuse_feat, _, _, _, logits, pred, _, _, _, _, _ = fix_model(
                    x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))

            # print("fuse feature mean and max:", fuse_feat.mean(), fuse_feat.max())
            # print("path feature mean and max:", path_feat.mean(), path_feat.max())

            # print("fuse logits:", F.softmax(logits[-1], 1))
            # print("path logits:", F.softmax(logit_path, 1))

            ### save the features and logits of all samples.
            fuse_features_all[index] = fuse_feat.cpu().detach().numpy()
            path_features_all[index] = path_feat.cpu().detach().numpy()

            fuse_preds_all[index] = F.softmax(logits[-1], 1).cpu().detach().numpy()
            path_preds_all[index] = F.softmax(logit_path, 1).cpu().detach().numpy()

            loss_cox = CoxLoss(survtime, censor, pred_path, device) if opt.task == "surv" else 0

            ### for the grading task, xiaohan added in 2021.12.22
            loss_cls = F.nll_loss(pred_path, grade) if opt.task == "grad" else 0
            if opt.num_teachers == 2:
                loss_div1, sample_loss_div1 = criterion_div(logit_path, logits[-1].detach())
                loss_div2, sample_loss_div2 = criterion_div(logit_path, ema_logit_path.detach())
                loss_div = loss_div1 + loss_div2
            elif opt.num_teachers == 1 and opt.which_teacher == "fuse":
                loss_div, sample_loss_div = criterion_div(logit_path, logits[-1].detach())
            elif opt.num_teachers == 1 and opt.which_teacher == "self_EMA":
                loss_div, sample_loss_div = criterion_div(logit_path, ema_logit_path.detach())

            ### xxh, 2022/12/27, assign weights to query samples according to 
            ### their discrepancy between the student and teacher model.
            # print(logit_path)
            # print(logits)
            teacher1_sample_weights = assign_sample_weights(F.softmax(logit_path, 1), F.softmax(logits[-1], 1), \
                                                            grade, opt.discrep_scale, opt.max_discrep)
            teacher2_sample_weights = assign_sample_weights(F.softmax(logit_path, 1), F.softmax(ema_logit_path, 1), \
                                                            grade, opt.discrep_scale, opt.max_discrep)

            # other kd beyond KL divergence
            if opt.distill == 'kd':
                loss_kd = 0
            elif opt.distill == "sp":
                loss_kd = criterion_sp(path_feat, fuse_feat.detach())
            elif opt.distill == 'crd':
                # print("contrast index:", sample_idx.shape)
                if epoch < opt.start_reweight:
                    teacher1_sample_weights = torch.ones_like(teacher1_sample_weights)
                    teacher2_sample_weights = torch.ones_like(teacher2_sample_weights)
                else:
                    teacher1_sample_weights += 1
                    teacher2_sample_weights += 1
                teacher1_sample_weights = teacher1_sample_weights.view(-1, 1)
                teacher1_all_sample_weights.append(teacher1_sample_weights.cpu().mean())
                    
                teacher2_sample_weights = teacher2_sample_weights.view(-1, 1)
                teacher2_all_sample_weights.append(teacher2_sample_weights.cpu().mean())

                if opt.num_teachers == 2:
                    loss_kd1, sample_loss_kd1 = criterion_kd(teacher1_sample_weights, path_feat, fuse_feat.detach(), grade, \
                                            index.cuda(), sample_idx.cuda())
                    loss_kd2, sample_loss_kd2 = criterion_kd_path(teacher2_sample_weights, path_feat, ema_path_feat.detach(), \
                                            grade, index.cuda(), sample_idx.cuda())
                    loss_kd = loss_kd1 + loss_kd2
                    # print("loss_kd1:", loss_kd1)

                elif opt.num_teachers == 1 and opt.which_teacher == "fuse":
                    loss_kd, sample_loss_kd = criterion_kd(teacher1_sample_weights, path_feat, fuse_feat.detach(), grade, \
                                            index.cuda(), sample_idx.cuda())
                elif opt.num_teachers == 1 and opt.which_teacher == "self_EMA":
                    loss_kd, sample_loss_kd = criterion_kd(teacher2_sample_weights, path_feat, ema_path_feat.detach(), \
                                            grade, index.cuda(), sample_idx.cuda())

            else:
                raise NotImplementedError(opt.distill)

            """
            根据各种loss的梯度一致性给不同的知识分配权重。
            """
            if opt.num_teachers == 2:
                loss_div1 = opt.alpha * sample_loss_div1
                loss_div2 = opt.alpha * sample_loss_div2

                
                if opt.distill == 'crd':
                    loss_kd1 = opt.beta * sample_loss_kd1
                    loss_kd2 = opt.beta * sample_loss_kd2
                    KD_loss_list = [loss_div1, loss_div2, loss_kd1, loss_kd2]
                    # print("KD_loss_list:", KD_loss_list)
                elif opt.distill == 'kd':
                    KD_loss_list = [sample_loss_div1, sample_loss_div2]


            if opt.assign_weights == "True":
                if opt.loss_weighting == "GK_refine":
                    scale, loss_KD = GK_refine_thresh(opt, optimizer, loss_cls, path_feat, KD_loss_list)
                    loss_weights += scale.detach().cpu().numpy()


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
            loss.backward()
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


        # ### compute the similarity matrices among all training samples. 2023/01/11
        # evaluate_feature(fuse_features_all, path_features_all, train_class_idx)
        # evaluate_logits(fuse_preds_all, path_preds_all, train_class_idx)

        print("loss scale:", loss_weights/len(train_loader))

        scheduler.step()

        if opt.distill == "crd":
            print("Average query weights for the multi-modal teacher:", np.mean(np.array(teacher1_all_sample_weights)))
            print("Average query weights for the self-EMA teacher:", np.mean(np.array(teacher2_all_sample_weights)))

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
        fuse_feat = fuse_feat.detach().cpu().numpy()
        feat_fuse_all = fuse_feat if feat_fuse_all is None else np.concatenate((feat_fuse_all, fuse_feat), axis=0)

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

    ### compute the similarity matrices among all testing samples. 2023/02/13
    evaluate_feature(feat_fuse_all, feat_path_all, gt_all)

    """compute other metrics for the grading task"""
    if opt.task == "grad":
        enc = LabelBinarizer()
        enc.fit(gt_all)
        grad_gt = enc.transform(gt_all)
        # print(grad_gt.shape, probs_all.shape)

        rocauc_fuse, ap_fuse, f1_micro_fuse, f1_gradeIV_fuse, f1_macro_fuse, \
            recall_macro_fuse, prec_macro_fuse, kappa_fuse, mcc_fuse = grading_metrics(grad_gt, probs_all)
        rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path, f1_macro_path, \
            recall_macro_path, prec_macro_path, kappa_path, mcc_path = grading_metrics(grad_gt, probs_path)
        print("fixed fuse branch:", rocauc_fuse, ap_fuse, f1_micro_fuse, f1_gradeIV_fuse)
        print("Path branch:", rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path)

        all_grad_metrics = [rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path, f1_macro_path, \
                            recall_macro_path, prec_macro_path, kappa_path, mcc_path]
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

    f1_micro = f1_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1), average=avg) ## 相当于accuracy和micro-recall
    f1_gradeIV = f1_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1), average=None)[2]
    print("f1-score:", f1_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1), average=None))

    f1_macro = f1_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1), average='macro')
    recall_macro = recall_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1), average='macro')
    precision_macro = precision_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1), average='macro')
    kappa = cohen_kappa_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1))
    mcc = matthews_corrcoef(np.argmax(y_label, axis=1), y_pred.argmax(axis=1))
    # print("kappa and mcc:", kappa, mcc)

    # print("confusion matrix:", confusion_matrix(np.argmax(y_label, axis=1), y_pred.argmax(axis=1)))

    return rocauc, ap, f1_micro, f1_gradeIV, f1_macro, recall_macro, precision_macro, kappa, mcc



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

        rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path, f1_macro_path, \
            recall_macro_path, prec_macro_path, kappa_path, mcc_path = grading_metrics(grad_gt, probs_path)
        print("Path branch:", rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path)
        all_grad_metrics = [rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path, f1_macro_path, \
                            recall_macro_path, prec_macro_path, kappa_path, mcc_path]
    else:
        all_grad_metrics = None

    ### save the predictions from the three branches and the ground truth in this list.
    pred_test = [risk_pred_all, risk_path_all, risk_omic_all, survtime_all, censor_all, \
        probs_all, probs_path, probs_omic, gt_all]
    grads_test = [grads_fuse_all, grads_path_all, grads_omic_all]
    feats_test = [feat_fuse_all, feat_path_all, feat_omic_all, gt_all]

    return loss_test, cindex_path, pvalue_test, surv_acc_test, grad_path_test, \
         all_grad_metrics, pred_test, grads_test, feats_test
