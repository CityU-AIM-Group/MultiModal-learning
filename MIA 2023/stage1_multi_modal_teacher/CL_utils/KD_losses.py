"""
Author: Xing Xiaohan
Data: 2021/12/15 11:13AM
"""
from cProfile import label
import torch
import torch.nn
from torch.nn import functional as F
import numpy as np


def pred_KD_loss(opt, p_s, p_t, labels = None, sample_KD="False"):
    """
    regularize the consistency between the model predictions of the teacher and student.
    for survival prediction task: 
        p_s, p_t: batch_size * 1
    for grading task:
        p_s, p_t: batch_size * num_classes
    """
    if opt.task == "surv":
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(p_s, p_t)

    elif opt.task == "grad":
        ### KL divergence for logits distillation. 
        assert p_s.size() == p_t.size()

        if sample_KD == "True":
            loss = torch.sum(F.kl_div(p_s, torch.exp(p_t), reduction='none'), axis=1)
        elif sample_KD == "False":
            loss = torch.sum(F.kl_div(p_s, torch.exp(p_t), reduction='none'))/p_s.shape[0]
            # loss = torch.sum(F.kl_div(p_s, torch.exp(p_t), reduction='none'), axis=1)
            # class_weights = torch.where(labels < 2, 1, 5)
            # loss = torch.sum(loss * class_weights)/p_s.shape[0]

        # print("pred KD loss:", loss)

    return loss


def SP_loss(f_s, f_t):
    """
    similarity preserving loss: constraint the similarity matrix of the teacher and student.
    f_s, f_t: batch_size * dim
    """
    f_t = f_t.clone().detach()
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1)

    G_s = torch.mm(f_s, torch.t(f_s))
    # G_s = G_s / G_s.norm(2)
    G_s = F.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    # G_t = G_t / G_t.norm(2)
    G_t = F.normalize(G_t)
    # print(G_s, G_t)

    G_diff = G_t - G_s
    weight = torch.abs(G_diff) + 1
    # weight = 2 * (weight - torch.mean(weight)) + 1
    # print(weight)
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    # loss = (weight * G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    # print("loss:", loss, "weighted loss:", weighted_loss)

    return loss