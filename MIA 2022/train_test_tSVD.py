"""
Author: Xing Xiaohan 
Date: 2021/12/31
在每个单模态和fuse模态之间构建4views，然后用t-SVD约束低秩。
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
from torch.utils.data import RandomSampler
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, accuracy_score

# from CL_utils.CRD_loss import CRDLoss
from CL_utils.CRD_criterion import CRDLoss
from CL_utils.orthogonal_loss import OrthLoss
# from CL_utils.CRD_loss import weighted_CRDLoss as CRDLoss
from CL_utils.KD_losses import pred_KD_loss
from networks_new import define_net, define_reg, define_optimizer, define_scheduler
from utils import CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, count_parameters, sigmoid_rampup
from data_loaders_MT import omic_transform

# from distiller_zoo import Similarity

from my_utils.TSVD_update_aux import update_aux

#from GPUtil import showUtilization as gpu_usage
import pdb
import pickle
import os


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def euclid_dist(x, y):
    x_sq = (x ** 2).mean(-1)
    x_sq_ = torch.stack([x_sq] * y.size(0), dim = 1)
    y_sq = (y ** 2).mean(-1)
    y_sq_ = torch.stack([y_sq] * x.size(0), dim = 0)
    xy = torch.mm(x, y.t()) / x.size(-1)
    dist = x_sq_ + y_sq_ - 2 * xy
    return dist        


#### newly added by Xiaohan, 2022/03/19
def update_adj_tensor(adj_tensor, feats):
    """
    construct adj matrix with pairwise cosine similarity.
    feats: [n_views, batch_size, dim]
    """
    for i in range(len(feats)):
        adj_matrix = torch.mm(feats[i], torch.t(feats[i]))
        adj_tensor[i] = F.normalize(adj_matrix)

        # ### 2022/05/12. 改成根据每两个样本之间的cosine similarity构建matrix.
        # feat = F.normalize(feats[i])
        # adj_tensor[i] = torch.mm(feat, torch.t(feat))
    # print(adj_tensor)
    return adj_tensor



#### newly added by Xiaohan, 2022/03/29
def update_triplet_adj_tensor(adj_tensor, feats):
    """
    construct adj matrix with triplet similarity.
    feats: [n_views, batch_size, dim]
    """
    for i in range(len(feats)):
        feats_diff = (2*feats[i].unsqueeze(0) - feats[i].unsqueeze(1)) # [bs, bs, dim]
        norm_diff = F.normalize(feats_diff, p=2, dim=2)
        adj_matrix = torch.bmm(norm_diff, norm_diff.transpose(1, 2)) # [bs, bs, bs]
        adj_matrix = adj_matrix.view(-1, adj_matrix.shape[-1])
        # print("adj matrix:", adj_matrix.grad_fn)
        adj_tensor[i] = F.normalize(adj_matrix)

    return adj_tensor


def tensor_nuclear_norm(tensor):
    tensor_TNN = 0
    tensor_rot = torch.stack(tensor).permute(1,2,0)
    for i in range(tensor_rot.shape[0]):
        tensor_TNN += torch.norm(tensor_rot[i], p='nuc')
    # print("tensor TNN:", tensor_TNN)


def train(opt, train_loader, n_data, test_loader, test_loader_patches, device, k):
    cudnn.deterministic = True
    seed = 2019 ### previous: 2019
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    def get_current_consistency_weight(epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return sigmoid_rampup(epoch, opt.consistency_rampup)

    def create_model(ema=False):
        model = define_net(opt, k).cuda()
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

    opt.n_data = n_data
    CRD_criterion_path = CRDLoss(opt).to(device)
    CRD_criterion_omic = CRDLoss(opt).to(device)
    CRD_criterion_fuse = CRDLoss(opt).to(device)

    ### define orthogonal loss.
    Orth_loss = OrthLoss().to(device)

    module_list = nn.ModuleList([])
    module_list.append(model)

    if opt.CRD_distill == 1:
        module_list.append(CRD_criterion_path.embed_s)
        module_list.append(CRD_criterion_path.embed_t)
        module_list.append(CRD_criterion_omic.embed_s)
        module_list.append(CRD_criterion_omic.embed_t)
        module_list.append(CRD_criterion_fuse.embed_s)
        module_list.append(CRD_criterion_fuse.embed_t)

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

    mu = opt.mu


    """newly added by Xiaohan, 2022/03/19, initialize the relation (adj) tensor."""
    adj_tensor1, aux_tensor1, adj_detach1 = list(), list(), list()
    adj_tensor2, aux_tensor2, adj_detach2 = list(), list(), list()
    prototypes1, prototypes2 = list(), list()

    for i in range(opt.n_views):
        adj_tensor1.append(torch.zeros(opt.batch_size, opt.batch_size).cuda())
        aux_tensor1.append(torch.zeros(opt.batch_size, opt.batch_size).cuda())
        adj_detach1.append(torch.zeros(opt.batch_size, opt.batch_size).cuda())
        
        adj_tensor2.append(torch.zeros(opt.batch_size, opt.batch_size).cuda())
        aux_tensor2.append(torch.zeros(opt.batch_size, opt.batch_size).cuda())
        adj_detach2.append(torch.zeros(opt.batch_size, opt.batch_size).cuda())
        
        prototypes1.append(torch.zeros(opt.label_dim, opt.feat_dim).cuda())
        prototypes2.append(torch.zeros(opt.label_dim, opt.feat_dim).cuda())


    for epoch in tqdm(range(opt.epoch_count, opt.niter+opt.niter_decay+1)):

        module_list.train()
        print("learning rate:", scheduler.get_lr())
        risk_pred_all, risk_path_all, risk_omic_all = np.array([]), np.array([]), np.array([])  # Used for calculating the C-Index
        censor_all, survtime_all = np.array([]), np.array([])
        loss_epoch, loss_fuse_epoch, loss_path_epoch, loss_omic_epoch = 0, 0, 0, 0
        grad_acc_epoch, grad_path_epoch, grad_omic_epoch = 0, 0, 0
        loss_CRD_epoch, loss_KD_epoch = 0, 0
        loss_tsvd_epoch = 0
        path_diff_epoch, omic_diff_epoch = 0, 0
        avg_path_TNN, avg_omic_TNN = 0, 0

        if epoch >= 15:
            opt.CRD_weight = 0.01

        for batch_idx, ((x_path, ema_x_path), x_grph, x_omic, censor, survtime, \
            grade, index, sample_idx) in enumerate(train_loader):

            censor = censor.to(device) if "surv" in opt.task else censor
            grade = grade.to(device) if "grad" in opt.task else grade

            # ema_x_omic = omic_transform(x_omic)
            # x_omic = omic_transform(x_omic)
            # print("sample index:", index)
            # print("contrastive samples:", sample_idx[:, :10])


            ema_x_omic = x_omic

            ### student model
            fuse_feat, path_feat, omic_feat, path_feat_f3, _, pred, pred_path, pred_omic, _, _, _ = model(
                x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))
            # print("features:", fuse_feat.shape, path_feat.shape, omic_feat.shape)
            # print("feature minimum:", torch.min(fuse_feat), torch.min(path_feat), torch.min(omic_feat))
            # print("predictions:", pred, pred_path, pred_omic)

            ### teacher model
            with torch.no_grad():
                ema_fuse_feat, ema_path_feat, ema_omic_feat, ema_path_feat_f3, _, ema_pred, ema_pred_path, \
                    ema_pred_omic, _, _, _ = ema_model(
                    x_path=ema_x_path.to(device), x_grph=x_grph.to(device), x_omic=ema_x_omic.to(device))

            # print(fuse_feat, ema_fuse_feat)

            loss_cox_path = CoxLoss(survtime, censor, pred_path, device) if opt.task == "surv" else 0
            loss_cox_omic = CoxLoss(survtime, censor, pred_omic, device) if opt.task == "surv" else 0
            loss_cox_fuse = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
            loss_cox = loss_cox_path + loss_cox_omic + loss_cox_fuse

            # KD_consistency_weight = get_current_consistency_weight(epoch)
            # KD_consistency_weight = 1.0

            if opt.CRD_distill == 1:
                # print("Using contrastive learning to distill the features.")
                # CRD_loss_path = CRD_criterion_path(path_feat, ema_path_feat, index.cuda(), sample_idx.cuda())
                # CRD_loss_omic = CRD_criterion_omic(omic_feat, ema_omic_feat, index.cuda(), sample_idx.cuda())
                # CRD_loss_fuse = CRD_criterion_fuse(fuse_feat, ema_fuse_feat, index.cuda(), sample_idx.cuda())
                # loss_CRD = opt.CRD_weight * (CRD_loss_path + CRD_loss_omic + CRD_loss_fuse) # / 3.0
                loss_CRD = opt.CRD_weight * CRD_criterion_fuse(
                    fuse_feat, ema_fuse_feat.detach(), index.cuda(), sample_idx.cuda())
                                
            else:
                loss_CRD = 0.0

            if opt.pred_distill == 1:
                # print("Distill the predictions.")
                # print(pred_path.shape, ema_pred_path.shape)
                if opt.num_teachers == 1:
                    pred_KD_fuse = pred_KD_loss(opt, pred, ema_pred)
                    pred_KD_path = pred_KD_loss(opt, pred_path, ema_pred_path)
                    pred_KD_omic = pred_KD_loss(opt, pred_omic, ema_pred_omic)

                ## each modality use the self and fuse teachers.
                elif opt.num_teachers == 2:
                    pred_KD_fuse = pred_KD_loss(opt, pred, ema_pred)
                    pred_KD_path = (pred_KD_loss(opt, pred_path, ema_pred_path) + pred_KD_loss(opt, pred_path, ema_pred))/2.0
                    pred_KD_omic = (pred_KD_loss(opt, pred_omic, ema_pred_omic) + pred_KD_loss(opt, pred_omic, ema_pred))/2.0

                ## each modality use the momentum encoders of the three branches as teachers.
                elif opt.num_teachers == 3:
                    pred_KD_fuse = pred_KD_loss(opt, pred, ema_pred)
                    pred_KD_path = (pred_KD_loss(opt, pred_path, ema_pred_path) + \
                        pred_KD_loss(opt, pred_path, ema_pred) + pred_KD_loss(opt, pred_path, ema_pred_omic))/3.0
                    pred_KD_omic = (pred_KD_loss(opt, pred_omic, ema_pred_omic) + \
                        pred_KD_loss(opt, pred_omic, ema_pred) + pred_KD_loss(opt, pred_omic, ema_pred_path))/3.0

                loss_pred_KD = opt.KD_weight * (pred_KD_fuse + pred_KD_path + pred_KD_omic) #/3.0
                # loss_pred_KD = opt.KD_weight * (pred_KD_fuse + pred_KD_path)/2.0
                # print("path:", pred_KD_path)
                # print("omic:", pred_KD_omic)
                # print("fuse:", pred_KD_fuse)


            else:
                loss_pred_KD = 0.0

            loss_reg = define_reg(opt, model)

            ### for the grading task, xiaohan added in 2021.12.22
            loss_nll_path = F.nll_loss(pred_path, grade) if opt.task == "grad" else 0
            loss_nll_omic = F.nll_loss(pred_omic, grade) if opt.task == "grad" else 0
            loss_nll_fuse = F.nll_loss(pred, grade) if opt.task == "grad" else 0
            loss_nll = loss_nll_path + loss_nll_omic + loss_nll_fuse

            loss = opt.lambda_cox*loss_cox + opt.lambda_nll*loss_nll + opt.lambda_reg*loss_reg \
                 + loss_CRD + loss_pred_KD

            if opt.orth_loss == "True":
                loss_orth = Orth_loss(path_feat, omic_feat)
                loss += loss_orth


            """
            Fix the model parameters, update the auxiliary variable: aux_tensor.
            Xiaohan, 2022/03/19
            """
            if opt.tSVD_loss == "True":
                # print("fuse feature:", torch.max(fuse_feat), torch.min(fuse_feat), torch.mean(fuse_feat))
                # print("path feature:", torch.max(path_feat), torch.min(path_feat), torch.mean(path_feat))
                # print("omic feature:", torch.max(omic_feat), torch.min(omic_feat), torch.mean(omic_feat))
                norm_fuse_feat = ema_fuse_feat/torch.max(ema_fuse_feat)
                norm_path_feat = ema_path_feat/torch.max(ema_path_feat)
                norm_omic_feat = ema_omic_feat/torch.max(ema_omic_feat)

                ### 对path模态的student和MT的不同层，以及fuse模态的detach特征用low-rank约束。
                if opt.n_views == 4:
                    ### path modality, 4 views
                    feats1 = [fuse_feat.detach(), ema_fuse_feat, path_feat, ema_path_feat]
                    ### omic modality, 4 views
                    feats2 = [fuse_feat.detach(), ema_fuse_feat, omic_feat, ema_omic_feat]

                    # feats1 = [fuse_feat, ema_fuse_feat, path_feat, ema_path_feat]
                    # feats2 = [fuse_feat, ema_fuse_feat, omic_feat, ema_omic_feat]

                elif opt.n_views == 2:
                    ### w/o fused view
                    feats1 = [path_feat, ema_path_feat]
                    feats2 = [omic_feat, ema_omic_feat]

                    # ### w/o mean-teacher view
                    # feats1 = [fuse_feat.detach(), path_feat]
                    # feats2 = [fuse_feat.detach(), omic_feat]

                ### 对不同模态之间的特征做mixup，构成更多的 views.
                elif opt.n_views == 6:
                    # ### path modality
                    # feats1 = [fuse_feat.detach(), ema_fuse_feat, path_feat, ema_path_feat,
                    #     0.2*norm_path_feat + 0.8*norm_fuse_feat, 0.4*norm_path_feat + 0.6*norm_fuse_feat]
                    # ### omic modality
                    # feats2 = [fuse_feat.detach(), ema_fuse_feat, omic_feat, ema_omic_feat,
                    #     0.2*norm_omic_feat + 0.8*norm_fuse_feat, 0.4*norm_omic_feat + 0.6*norm_fuse_feat]

                    ### path modality
                    feats1 = [fuse_feat.detach(), ema_fuse_feat, path_feat, ema_path_feat,
                        0.9*norm_path_feat + 0.1*norm_omic_feat, 0.8*norm_path_feat + 0.2*norm_omic_feat]
                    ### omic modality
                    feats2 = [fuse_feat.detach(), ema_fuse_feat, omic_feat, ema_omic_feat,
                        0.9*norm_omic_feat + 0.1*norm_path_feat, 0.8*norm_omic_feat + 0.2*norm_path_feat]

                elif opt.n_views == 8:
                    # ### path modality
                    # feats1 = [fuse_feat.detach(), ema_fuse_feat, path_feat, ema_path_feat,
                    #     0.2*norm_path_feat + 0.8*norm_fuse_feat, 0.4*norm_path_feat + 0.6*norm_fuse_feat,
                    #     0.6*norm_path_feat + 0.4*norm_fuse_feat, 0.8*norm_path_feat + 0.2*norm_fuse_feat]
                    # ### omic modality
                    # feats2 = [fuse_feat.detach(), ema_fuse_feat, omic_feat, ema_omic_feat,
                    #     0.2*norm_omic_feat + 0.8*norm_fuse_feat, 0.4*norm_omic_feat + 0.6*norm_fuse_feat,
                    #     0.6*norm_omic_feat + 0.4*norm_fuse_feat, 0.8*norm_omic_feat + 0.2*norm_fuse_feat]

                    ### path modality
                    feats1 = [fuse_feat.detach(), ema_fuse_feat, path_feat, ema_path_feat,
                        0.9*norm_path_feat + 0.1*norm_omic_feat, 0.8*norm_path_feat + 0.2*norm_omic_feat,
                        0.7*norm_path_feat + 0.3*norm_omic_feat, 0.6*norm_path_feat + 0.4*norm_omic_feat]
                    ### omic modality
                    feats2 = [fuse_feat.detach(), ema_fuse_feat, omic_feat, ema_omic_feat,
                        0.9*norm_omic_feat + 0.1*norm_path_feat, 0.8*norm_omic_feat + 0.2*norm_path_feat,
                        0.7*norm_omic_feat + 0.3*norm_path_feat, 0.6*norm_omic_feat + 0.4*norm_path_feat]


                adj_tensor1 = update_adj_tensor(adj_tensor1, feats1)
                adj_tensor2 = update_adj_tensor(adj_tensor2, feats2)

                # # print("prototype1:", prototypes1)
                # adj_tensor1, prototypes1 = update_adj_tensor(opt, adj_tensor1, feats1, grade, prototypes1)
                # adj_tensor2, prototypes2 = update_adj_tensor(opt, adj_tensor2, feats2, grade, prototypes2)

                # adj_tensor1 = update_triplet_adj_tensor(adj_tensor1, feats1)
                # adj_tensor2 = update_triplet_adj_tensor(adj_tensor2, feats2)

                for view_idx in range(opt.n_views):
                    adj_detach1[view_idx] = adj_tensor1[view_idx].detach()
                    adj_detach2[view_idx] = adj_tensor2[view_idx].detach()

                if batch_idx%opt.aux_iter == 0:
                    print_bool = True if batch_idx is 0 else False
                    # print("tSVD mode:", opt.tSVD_mode)
                    if opt.tSVD_mode == "path" or "pathomic":
                        # print("tSVD on the path modality")
                        # update auxiliary variable
                        adj = torch.stack(adj_detach1, dim=2)
                        # print("fuse mean relation:", adj[0].mean())
                        # print("fuse_ema mean relation:", adj[1].mean())
                        # print("path mean relation:", adj[2].mean())
                        # print("path_ema mean relation:", adj[3].mean())
                        aux, path_TNN = update_aux(adj, opt.Lambda_global / mu, print_bool)
                        # aux, path_TNN = update_aux(adj, opt.Lambda_global, print_bool)
                        aux = list(torch.split(aux, 1, dim=2))
                        for view_idx in range(opt.n_views):
                            aux_tensor1[view_idx] = aux[view_idx].squeeze().float().cuda()
                        avg_path_TNN += path_TNN
                        # print("recovered fuse mean relation:", aux[0].mean())
                        # print("recovered fuse_ema mean relation:", aux[1].mean())
                        # print("recovered path mean relation:", aux[2].mean())
                        # print("recovered path_ema mean relation:", aux[3].mean())

                        # print('[{:s}]\t\t path g-TNN: {:.4f}'.format('Train', path_TNN))

                    if opt.tSVD_mode == "omic" or "pathomic":
                        # print("tSVD on the omic modality")
                        # update auxiliary variable
                        adj = torch.stack(adj_detach2, dim=2)
                        aux, omic_TNN = update_aux(adj, opt.Lambda_global / mu, print_bool)
                        # aux, omic_TNN = update_aux(adj, opt.Lambda_global, print_bool)
                        aux = list(torch.split(aux, 1, dim=2))
                        for view_idx in range(opt.n_views):
                            aux_tensor2[view_idx] = aux[view_idx].squeeze().float().cuda()
                        avg_omic_TNN += omic_TNN
                        # print('[{:s}]\t\t omic g-TNN: {:.4f}'.format('Train', omic_TNN))

                    # update parameter mu
                    mu = min(mu * opt.pho, opt.max_mu)
                    # print("updated mu:", opt.mu)


            ### update the adj_tensor and compute tSVD loss
            if opt.tSVD_loss == "True":                
                loss_tsvd = 0
                for view_idx in range(opt.n_views):
                    if opt.tSVD_mode == "path":
                        # print("path TSVD")
                        loss_tsvd += mu/2.0 * (torch.norm(adj_tensor1[view_idx]-aux_tensor1[view_idx]))**2
                    elif opt.tSVD_mode == "omic":
                        loss_tsvd += mu/2.0 * (torch.norm(adj_tensor2[view_idx]-aux_tensor2[view_idx]))**2
                    elif opt.tSVD_mode == "pathomic":
                        # print("pathomic TSVD")
                        loss_tsvd += (mu/2.0 * ((torch.norm(adj_tensor1[view_idx]-aux_tensor1[view_idx]))**2 \
                            + (torch.norm(adj_tensor2[view_idx]-aux_tensor2[view_idx]))**2))

                loss += loss_tsvd

                # path_diff = torch.mean(torch.abs(adj_tensor1[0] + adj_tensor1[1] + adj_tensor1[3] - 3*adj_tensor1[2]))
                # omic_diff = torch.mean(torch.abs(adj_tensor2[0] + adj_tensor2[1] + adj_tensor2[3] - 3*adj_tensor2[2]))

                # path_diff_epoch += path_diff.item()
                # omic_diff_epoch += omic_diff.item()


                # ### compute tensor nuclear norm (TNN)
                # tensor_nuclear_norm(adj_tensor1)


                # # print("tsvd loss:", loss_tsvd)
                # adj = return_adj1
                # adj.grad = None
                # adj.retain_grad()
                # loss_tsvd.backward(retain_graph=True)
                # print("adj grad:", adj.grad)


            else:
                loss_tsvd = 0

            loss_epoch += loss.data.item()
            loss_fuse_epoch += (loss_cox_fuse + loss_nll_fuse).item()
            loss_path_epoch += (loss_cox_path + loss_nll_path).item()
            loss_omic_epoch += (loss_cox_omic + loss_nll_omic).item()

            if loss_tsvd > 0.0:
                loss_tsvd_epoch += loss_tsvd.item()

            if loss_CRD > 0.0:
                loss_CRD_epoch += loss_CRD.item()

            if loss_pred_KD > 0.0:
                loss_KD_epoch += loss_pred_KD.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, opt.ema_decay, iter_num)
            iter_num = iter_num + 1

            if opt.lr_policy == "onecycle":
                scheduler.step()

            if opt.task == "surv":
                risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
                risk_path_all = np.concatenate((risk_path_all, pred_path.detach().cpu().numpy().reshape(-1)))
                risk_omic_all = np.concatenate((risk_omic_all, pred_omic.detach().cpu().numpy().reshape(-1)))
                censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
                survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information
            
            elif opt.task == "grad":
                pred = pred.argmax(dim=1, keepdim=True)
                grad_acc_epoch += pred.eq(grade.view_as(pred)).sum().item()
                pred_path = pred_path.argmax(dim=1, keepdim=True)
                grad_path_epoch += pred_path.eq(grade.view_as(pred_path)).sum().item()
                pred_omic = pred_omic.argmax(dim=1, keepdim=True)
                grad_omic_epoch += pred_omic.eq(grade.view_as(pred_omic)).sum().item()

            if opt.verbose > 0 and opt.print_every > 0 and (batch_idx % opt.print_every == 0 or batch_idx+1 == len(train_loader)):
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch+1, opt.niter+opt.niter_decay, batch_idx+1, len(train_loader), loss.item()))


        # print("original relation matrix:", torch.mean(adj_tensor1[0]), torch.mean(adj_tensor1[1]), \
        #     torch.mean(adj_tensor1[2]), torch.mean(adj_tensor1[3]))
        # print("aux relation matrix:", torch.mean(aux_tensor1[0]), torch.mean(aux_tensor1[1]), \
        #     torch.mean(aux_tensor1[2]), torch.mean(aux_tensor1[3]))

        # print("original relation matrix:", adj_tensor1[0])


        if opt.lr_policy != "onecycle":
            scheduler.step()

        if opt.measure or epoch == (opt.niter+opt.niter_decay - 1):
            loss_epoch /= len(train_loader)
            loss_fuse_epoch /= len(train_loader)
            loss_path_epoch /= len(train_loader)
            loss_omic_epoch /= len(train_loader)
            loss_CRD_epoch /= len(train_loader)
            loss_KD_epoch /= len(train_loader)
            loss_tsvd_epoch /= len(train_loader)
            path_diff_epoch /= len(train_loader)
            omic_diff_epoch /= len(train_loader)

            avg_path_TNN /= len(train_loader)
            avg_omic_TNN /= len(train_loader)

            cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
            cindex_path = CIndex_lifeline(risk_path_all, censor_all, survtime_all) if opt.task == 'surv' else None
            cindex_omic = CIndex_lifeline(risk_omic_all, censor_all, survtime_all) if opt.task == 'surv' else None

            ### histological grading.
            grad_acc_epoch = grad_acc_epoch / len(train_loader.dataset) if opt.task == 'grad' else None
            grad_path_epoch = grad_path_epoch / len(train_loader.dataset) if opt.task == 'grad' else None
            grad_omic_epoch = grad_omic_epoch / len(train_loader.dataset) if opt.task == 'grad' else None

            ### test the model. 最后的15个周期用全部patches测试模型。
            if epoch > (opt.niter+opt.niter_decay - 15):
                test_loader = test_loader_patches
            loss_test, loss_test_fuse, loss_test_path, loss_test_omic, cindex_test, cindex_test_path, cindex_test_omic, \
                pvalue_test, surv_acc_test, grad_acc_test, grad_path_test, grad_omic_test, _, pred_test, _, _ = test(
                    opt, module_list, model, test_loader, device)

            if opt.task == "surv": grad_acc_test = cindex_test
            if epoch > 15 and grad_acc_test > best_acc:
                best_acc = grad_acc_test
                save_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, \
                    '%s_%d_best.pt' % (opt.model_name, k))
                model_state_dict = model.module.state_dict()
                ema_model_state_dict = ema_model.module.state_dict()
                torch.save({
                    'split':k,
                    'opt': opt,
                    'epoch': opt.niter+opt.niter_decay,
                    'model_state_dict': model_state_dict,
                    'ema_model_state_dict': ema_model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metric_logger}, 
                    save_path)
                print("saving the model at:", save_path)

            # pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%s%d_pred_test.pkl' % (opt.model_name, k, use_patch, epoch)), 'wb'))

            if opt.verbose > 0:
                if opt.task == 'surv':

                    print('[{:s}]\t\t CRD Loss: {:.4f}'.format('Train', loss_CRD_epoch))
                    print('[{:s}]\t\t pred KD Loss: {:.4f}'.format('Train', loss_KD_epoch))
                    print('[{:s}]\t\t tSVD Loss: {:.4f}'.format('Train', loss_tsvd_epoch))
                    # print('[{:s}]\t\t path g-TNN: {:.4f}'.format('Train', avg_path_TNN))
                    # print('[{:s}]\t\t omic g-TNN: {:.4f}'.format('Train', avg_omic_TNN))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_path_epoch, 'path C-Index', cindex_path))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_omic_epoch, 'omic C-Index', cindex_omic))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_fuse_epoch, 'fusion C-Index', cindex_epoch))

                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Test', loss_test_path, 'path C-Index', cindex_test_path))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Test', loss_test_omic, 'omic C-Index', cindex_test_omic))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Test', loss_test_fuse, 'fusion C-Index', cindex_test))

                elif opt.task == 'grad':
                    print('[{:s}]\t\t CRD Loss: {:.4f}'.format('Train', loss_CRD_epoch))
                    print('[{:s}]\t\t pred KD Loss: {:.4f}'.format('Train', loss_KD_epoch))
                    print('[{:s}]\t\t tSVD Loss: {:.4f}'.format('Train', loss_tsvd_epoch))
                    print('[{:s}]\t\t path g-TNN: {:.4f}'.format('Train', avg_path_TNN))
                    print('[{:s}]\t\t omic g-TNN: {:.4f}'.format('Train', avg_omic_TNN))
                    # print('[{:s}]\t\t path cross-modal difference: {:.4f}'.format('Train', path_diff_epoch))
                    # print('[{:s}]\t\t omic cross-modal difference: {:.4f}'.format('Train', omic_diff_epoch))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_path_epoch, 'path accuracy', grad_path_epoch))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_omic_epoch, 'omic accuracy', grad_omic_epoch))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_fuse_epoch, 'fuse accuracy', grad_acc_epoch))

                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test_path, 'path accuracy', grad_path_test))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test_omic, 'omic accuracy', grad_omic_test))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test_fuse, 'fuse accuracy', grad_acc_test))


    return module_list, model, ema_model, optimizer, metric_logger


def test(opt, module_list, model, test_loader, device):
    module_list.eval()
    risk_pred_all, risk_path_all, risk_omic_all = np.array([]), np.array([]), np.array([])  # Used for calculating the C-Index
    censor_all, survtime_all = np.array([]), np.array([])

    probs_all, probs_path, probs_omic, gt_all = None, None, None, np.array([])
    loss_test, loss_fuse_test, loss_path_test, loss_omic_test = 0, 0, 0, 0
    grad_acc_test, grad_path_test, grad_omic_test = 0, 0, 0
    grads_fuse_all, grads_path_all, grads_omic_all = None, None, None
    feat_fuse_all, feat_path_all, feat_omic_all = None, None, None

    for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade) in enumerate(test_loader):

        censor = censor.to(device) if "surv" in opt.task else censor
        grade = grade.to(device) if "grad" in opt.task else grade

        feat_fuse, feat_path, feat_omic, _, _, pred, pred_path, pred_omic, grads_fuse, grads_path, grads_omic = model(
            x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))

        # print("predicted hazards:", pred, pred_path, pred_omic)
        # print("fuse:", torch.exp(pred))
        # print("path:", torch.exp(pred_path))
        # print("omic:", torch.exp(pred_omic))
        # print("fuse features:", torch.max(feat_fuse), torch.min(feat_fuse), torch.mean(feat_fuse))
        # print("path features:", torch.max(feat_path), torch.min(feat_path), torch.mean(feat_path))
        # print("omic features:", torch.max(feat_omic), torch.min(feat_omic), torch.mean(feat_omic))

        loss_cox_fuse = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
        loss_cox_path = CoxLoss(survtime, censor, pred_path, device) if opt.task == "surv" else 0
        loss_cox_omic = CoxLoss(survtime, censor, pred_omic, device) if opt.task == "surv" else 0
        loss_cox = loss_cox_fuse + loss_cox_path + loss_cox_omic

        loss_reg = define_reg(opt, model)

        ### for the grading task, xiaohan added in 2021.12.09
        loss_nll_fuse = F.nll_loss(pred, grade) if opt.task == "grad" else 0
        loss_nll_path = F.nll_loss(pred_path, grade) if opt.task == "grad" else 0
        loss_nll_omic = F.nll_loss(pred_omic, grade) if opt.task == "grad" else 0
        loss_nll = loss_nll_fuse + loss_nll_path + loss_nll_omic

        loss = opt.lambda_cox*loss_cox + opt.lambda_nll*loss_nll + opt.lambda_reg*loss_reg
        loss_test += loss.data.item()
        loss_fuse_test += (loss_cox_fuse + loss_nll_fuse).item()
        loss_path_test += (loss_cox_path + loss_nll_path).item()
        loss_omic_test += (loss_cox_omic + loss_nll_omic).item()

        gt_all = np.concatenate((gt_all, grade.detach().cpu().numpy().reshape(-1)))   # Logging Information

        feat_fuse, feat_path, feat_omic = feat_fuse.detach().cpu().numpy(), \
            feat_path.detach().cpu().numpy(), feat_omic.detach().cpu().numpy()
        feat_fuse_all = feat_fuse if feat_fuse_all is None else np.concatenate((feat_fuse_all, feat_fuse), axis=0)
        feat_path_all = feat_path if feat_path_all is None else np.concatenate((feat_path_all, feat_path), axis=0)
        feat_omic_all = feat_omic if feat_omic_all is None else np.concatenate((feat_omic_all, feat_omic), axis=0)

        if opt.return_grad == "True":
            grads_fuse_all = grads_fuse if grads_fuse_all is None else np.concatenate((grads_fuse_all, grads_fuse), axis=0)
            grads_path_all = grads_path if grads_path_all is None else np.concatenate((grads_path_all, grads_path), axis=0)
            grads_omic_all = grads_omic if grads_omic_all is None else np.concatenate((grads_omic_all, grads_omic), axis=0)


        if opt.task == "surv":
            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
            risk_path_all = np.concatenate((risk_path_all, pred_path.detach().cpu().numpy().reshape(-1)))
            risk_omic_all = np.concatenate((risk_omic_all, pred_omic.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information
        elif opt.task == "grad":
            grad_acc_test, probs_all = compute_accuracy(pred, grade, probs_all, grad_acc_test)
            grad_path_test, probs_path = compute_accuracy(pred_path, grade, probs_path, grad_path_test)
            grad_omic_test, probs_omic = compute_accuracy(pred_omic, grade, probs_omic, grad_omic_test)

    ################################################### 
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader)
    loss_fuse_test /= len(test_loader)
    loss_path_test /= len(test_loader)
    loss_omic_test /= len(test_loader)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    cindex_path = CIndex_lifeline(risk_path_all, censor_all, survtime_all) if opt.task == 'surv' else None
    cindex_omic = CIndex_lifeline(risk_omic_all, censor_all, survtime_all) if opt.task == 'surv' else None
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all) if opt.task == 'surv' else None

    grad_acc_test = grad_acc_test / len(test_loader.dataset) if opt.task == 'grad' else None
    grad_path_test = grad_path_test / len(test_loader.dataset) if opt.task == 'grad' else None
    grad_omic_test = grad_omic_test / len(test_loader.dataset) if opt.task == 'grad' else None

    """compute other metrics for the grading task"""
    if opt.task == "grad":
        # enc = LabelBinarizer()
        # enc.fit(gt_all)
        # grad_gt = enc.transform(gt_all)

        grad_gt = torch.zeros(gt_all.shape[0], opt.label_dim).scatter_(1, torch.LongTensor(gt_all).view(-1,1), 1)
        # print(gt_all.shape, grad_gt.shape, probs_all.shape)

        rocauc_fuse, ap_fuse, f1_micro_fuse, f1_gradeIV_fuse = grading_metrics(grad_gt, probs_all)
        rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path = grading_metrics(grad_gt, probs_path)
        rocauc_omic, ap_omic, f1_micro_omic, f1_gradeIV_omic = grading_metrics(grad_gt, probs_omic)
        print("Fused branch:", rocauc_fuse, ap_fuse, f1_micro_fuse, f1_gradeIV_fuse)
        print("Path branch:", rocauc_path, ap_path, f1_micro_path, f1_gradeIV_path)
        print("Omic branch:", rocauc_omic, ap_omic, f1_micro_omic, f1_gradeIV_omic)

        all_grad_metrics = [rocauc_fuse, ap_fuse, f1_micro_fuse, f1_gradeIV_fuse, rocauc_path, ap_path, \
            f1_micro_path, f1_gradeIV_path, rocauc_omic, ap_omic, f1_micro_omic, f1_gradeIV_omic]
    else:
        all_grad_metrics = None

    ### save the predictions from the three branches and the ground truth in this list.
    pred_test = [risk_pred_all, risk_path_all, risk_omic_all, survtime_all, censor_all, \
        probs_all, probs_path, probs_omic, gt_all]
    grads_test = [grads_fuse_all, grads_path_all, grads_omic_all]
    feats_test = [feat_fuse_all, feat_path_all, feat_omic_all, gt_all]

    return loss_test, loss_fuse_test, loss_path_test, loss_omic_test, cindex_test, cindex_path, cindex_omic, pvalue_test, \
        surv_acc_test, grad_acc_test, grad_path_test, grad_omic_test, all_grad_metrics, pred_test, grads_test, feats_test


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
    # print(y_pred)
    rocauc = roc_auc_score(y_label, y_pred, average = avg)
    ap = average_precision_score(y_label, y_pred, average=avg)
    # f1_micro = f1_score(y_pred.argmax(axis=1), np.argmax(y_label, axis=1), average=avg)
    # f1_gradeIV = f1_score(y_pred.argmax(axis=1), np.argmax(y_label, axis=1), average=None)[2]
    # print("f1-score:", f1_score(y_pred.argmax(axis=1), np.argmax(y_label, axis=1), average=None))

    f1_micro = f1_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1), average=avg)
    f1_gradeIV = f1_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1), average=None)[2]
    # print("f1-score:", f1_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1), average=None))

    # print("accuracy:", accuracy_score(np.argmax(y_label, axis=1), y_pred.argmax(axis=1)))
    # print("confusion matrix:", confusion_matrix(np.argmax(y_label, axis=1), y_pred.argmax(axis=1)))

    return rocauc, ap, f1_micro, f1_gradeIV
