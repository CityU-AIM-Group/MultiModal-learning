import torch
from torch import nn
from .memory_new import ContrastMemory_v3

eps = 1e-7

class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side
    """
    def __init__(self, opt, n_data):
        super(CRDLoss, self).__init__()
        self.P = opt.nce_p
        self.P2 = opt.nce_p2
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)

        self.contrast = ContrastMemory_v3(opt.feat_dim, n_data, opt.nce_p, opt.nce_k, \
            opt.nce_t, opt.nce_m, opt.select_pos_pairs, opt.nce_p2, opt.select_neg_pairs, opt.nce_k2)
        self.criterion_t = ContrastLoss_v2(n_data, sample_KD=opt.sample_KD)
        self.criterion_s = ContrastLoss_v2(n_data, sample_KD=opt.sample_KD)

        self.select_pos_mode = opt.select_pos_mode

    def forward(self, epoch, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        # out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        out_s, out_t = self.contrast(epoch, f_s, f_t, idx, contrast_idx, self.select_pos_mode)
        # s_loss = self.criterion_s(out_s)
        # t_loss = self.criterion_t(out_t)
        # s_loss = self.criterion_s(out_s, self.P)
        # t_loss = self.criterion_t(out_t, self.P)
        s_loss = self.criterion_s(out_s, self.P2)
        t_loss = self.criterion_t(out_t, self.P2)
        loss = s_loss + t_loss
        return loss


class ContrastLoss_v2(nn.Module):
    """
    supervised contrastive loss.
    """
    def __init__(self, n_data, sample_KD):
        super(ContrastLoss_v2, self).__init__()
        self.n_data = n_data
        self.sample_KD = sample_KD

    def forward(self, x, P):
        bsz = x.shape[0]
        N = x.size(1) - P
        m = N
        # print(P, N)

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.narrow(1, 0, P)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()
        # print("positive:", log_D1.shape)

        # loss for K negative pair
        P_neg = x.narrow(1, P, N)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()
        # print("negative:", log_D0.shape)

        if self.sample_KD == "False":
            ### new version: 2021 Feb. (Using positive samples from the memory bank)
            ### average of the 1 exact pos. sample and (P-1) relax pos. samples.
            loss = - ((log_D1.squeeze().sum(0) + log_D0.view(-1, 1).repeat(1, P).sum(0)) / bsz).sum(0) / P
            # print((log_D1.squeeze().sum(0) + log_D0.view(-1, 1).repeat(1, P).sum(0)).shape)

        elif self.sample_KD == "True":
            loss = - ((log_D1.squeeze(-1) + (log_D0.repeat(1, 1, P)).sum(1))).sum(1) / P
            # print((log_D1.squeeze(-1) + (log_D0.repeat(1, 1, P)).sum(1)).shape)
            # print(log_D1.squeeze(-1).shape)
            # print(log_D0.repeat(1, 1, P).sum(1).shape)

        return loss



class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
