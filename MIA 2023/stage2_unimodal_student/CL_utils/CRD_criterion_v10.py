"""
Author: Xiaohan Xing
Date: 2023/02/04
尝试以下几种情况：
1. 计算query samples的memory feature和其他所有样本之间的memory features相似度选择最相似的P个样本作为正样本。
将P个正样本对应的CRD loss按照它们和query之间的相似度加权 (设置num_neighbors = 3或5)
2. 对每个类别的样本求平均值 (class center)或者P个聚类中心。
将query同类别的centers作为正样本对, 不同类别的centers作为负样本对。
"""

import torch
from torch import nn
import math
from sklearn.cluster import KMeans
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

eps = 1e-7


class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    return out_v1, out_v2: [batch size, K+num_pos, 1]
    """
    def __init__(self, inputSize, outputSize, train_class_idx, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.K = K
        self.class_idx = train_class_idx

        self.all_sample_labels = torch.zeros(outputSize)
        for c in range(len(self.class_idx)):
            class_c_idx = self.class_idx[c]
            for i in range(class_c_idx.shape[0]):
                self.all_sample_labels[class_c_idx[i]] = c

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, num_pos, pos_extra, v1, v2, batch_label, y, idx=None):
        # print("query index and label:", y, batch_label)
        # print("query label:", batch_label)
        # print("contrast idx:", idx)
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        class_mask = torch.zeros(len(self.class_idx), outputSize).cuda()
        for c in range(len(self.class_idx)):
            class_mask[c] = torch.where(self.all_sample_labels==c, 1, 0)
        batch_class_mask = torch.index_select(class_mask, 0, batch_label)

        ### 每个query样本不属于哪个类别,之后取这些类别的centers作为negative pairs.
        onehot_label = F.one_hot(batch_label, num_classes=3).cpu().detach().numpy()
        batch_neg_labels = torch.tensor(np.argwhere(onehot_label==0)[:, 1]).cuda() ## [batch_size*(num_classes-1)]

        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        if pos_extra == "neighbors":
            v1_similarity = batch_class_mask * torch.tensor(cosine_similarity(weight_v1[:, 0, :].detach().cpu().numpy(), \
                                self.memory_v1.detach().cpu().numpy())).cuda() 
            v1_neighbors = torch.sort(v1_similarity, descending=True, dim=-1)[1][:, :num_pos].cuda()
            # neighbor_labels = torch.index_select(self.all_sample_labels.cuda(), 0, v1_neighbors.reshape(-1)).view(batchSize, num_pos)
            # print("neighbors index and labels:", v1_neighbors, neighbor_labels)
            v1_neighbors_similarity = torch.sort(v1_similarity, descending=True, dim=-1)[0][:, :num_pos]
            v1_knn_pos = torch.index_select(self.memory_v1, 0, v1_neighbors.reshape(-1)).detach()
            v1_knn_pos = v1_knn_pos.view(batchSize, num_pos, inputSize)
            weight_v1 = torch.cat((v1_knn_pos, weight_v1[:, 1:, :]), 1)
        elif pos_extra == "centers":
            ### 除了一个query自身之外，再取(num_pos-1)个同类别centers作为正样本, 以及不同类别的centers作为负样本. 
            class_centers_v1 = torch.zeros(len(self.class_idx), num_pos-1, inputSize).cuda()
            ### 对每个类别做kmeans clustering, 求出k个cluster centers.
            for c in range(len(self.class_idx)):
                feature_v1 = torch.index_select(self.memory_v1, 0, torch.tensor(self.class_idx[c]).cuda()).detach()
                if num_pos == 2:
                    class_centers_v1[c] = torch.mean(feature_v1, 0)
                elif num_pos > 2:
                    estimator = KMeans(n_clusters = num_pos-1)
                    estimator.fit(feature_v1.cpu().numpy())
                    class_centers_v1[c] = torch.tensor(estimator.cluster_centers_).cuda()

            ### select the features of class centers for each sample.
            weight_v1_centers = torch.index_select(class_centers_v1, 0, batch_label).detach()
            weight_v1_centers = weight_v1_centers.view(batchSize, num_pos-1, inputSize)
            weight_v1 = torch.cat((weight_v1_centers, weight_v1), 1)
            ### 取不同类别的centers作为负样本对
            weight_v1_neg_centers = torch.index_select(class_centers_v1, 0, batch_neg_labels).detach()
            weight_v1_neg_centers = weight_v1_neg_centers.view(batchSize, 2*(num_pos-1), inputSize)
            weight_v1 = torch.cat((weight_v1, weight_v1_neg_centers), 1)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))

        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        if pos_extra == "neighbors":
            ## 取出来的neighbors中的第一个index一定是query样本自身
            v2_similarity = batch_class_mask * torch.tensor(cosine_similarity(weight_v2[:, 0, :].detach().cpu().numpy(), \
                                self.memory_v2.detach().cpu().numpy())).cuda()        
            v2_neighbors = torch.sort(v2_similarity, descending=True, dim=-1)[1][:, :num_pos].cuda()
            v2_neighbors_similarity = torch.sort(v2_similarity, descending=True, dim=-1)[0][:, :num_pos]
            v2_knn_pos = torch.index_select(self.memory_v2, 0, v2_neighbors.reshape(-1)).detach()
            v2_knn_pos = v2_knn_pos.view(batchSize, num_pos, inputSize)
            weight_v2 = torch.cat((v2_knn_pos, weight_v2[:, 1:, :]), 1)
        elif pos_extra == "centers":
            ### 除了一个query自身之外，再取(num_pos-1)个同类别centers作为正样本, 以及不同类别的centers作为负样本. 
            class_centers_v2 = torch.zeros(len(self.class_idx), num_pos-1, inputSize).cuda()
            ### 对每个类别做kmeans clustering, 求出k个cluster centers.
            for c in range(len(self.class_idx)):
                feature_v2 = torch.index_select(self.memory_v2, 0, torch.tensor(self.class_idx[c]).cuda()).detach()
                if num_pos == 2:
                    class_centers_v2[c] = torch.mean(feature_v2, 0)
                elif num_pos > 2:
                    estimator = KMeans(n_clusters = num_pos-1)
                    estimator.fit(feature_v2.cpu().numpy())
                    class_centers_v2[c] = torch.tensor(estimator.cluster_centers_).cuda()

            ### select the features of class centers for each sample.
            weight_v2_centers = torch.index_select(class_centers_v2, 0, batch_label).detach()
            weight_v2_centers = weight_v2_centers.view(batchSize, num_pos-1, inputSize)
            weight_v2 = torch.cat((weight_v2_centers, weight_v2), 1)        
            ### 取不同类别的centers作为负样本对
            weight_v2_neg_centers = torch.index_select(class_centers_v2, 0, batch_neg_labels).detach()
            weight_v2_neg_centers = weight_v2_neg_centers.view(batchSize, 2*(num_pos-1), inputSize)
            weight_v2 = torch.cat((weight_v2, weight_v2_neg_centers), 1)

        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        # print("out_v1 and out_v2:", out_v1.shape, out_v2.shape)
        if pos_extra == "neighbors":
            return out_v1, out_v2, v1_neighbors_similarity, v2_neighbors_similarity
        else:
            return out_v1, out_v2
        


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt, n_data, train_class_idx):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        # print("n_data:", opt.n_data)
        self.contrast = ContrastMemory(opt.feat_dim, n_data, train_class_idx, opt.nce_k, opt.nce_t, opt.nce_m)
        self.num_pos = opt.nce_p
        self.pos_extra = opt.pos_extra

        if self.pos_extra == "neighbors":
            self.criterion_t = ContrastLoss_v2(n_data)
            self.criterion_s = ContrastLoss_v2(n_data)
        else:
            self.criterion_t = ContrastLoss(n_data)
            self.criterion_s = ContrastLoss(n_data)

    def forward(self, sample_weights, f_s, f_t, batch_label, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        # print("teacher feature:", f_t.shape, torch.max(f_t), torch.min(f_t), torch.mean(f_t))
        # print("student feature:", f_s.shape, torch.max(f_s), torch.min(f_s), torch.mean(f_s))
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        if self.pos_extra == "neighbors":
            out_s, out_t, s_similarity, t_similarity = self.contrast(
                self.num_pos, self.pos_extra, f_s, f_t, batch_label, idx, contrast_idx)
            s_loss, s_sample_loss = self.criterion_s(sample_weights, out_s, self.num_pos, t_similarity)
            t_loss, t_sample_loss = self.criterion_t(sample_weights, out_t, self.num_pos, s_similarity)
        else:
            out_s, out_t = self.contrast(self.num_pos, self.pos_extra, f_s, f_t, batch_label, idx, contrast_idx)
            s_loss, s_sample_loss = self.criterion_s(sample_weights, out_s, self.num_pos)
            t_loss, t_sample_loss = self.criterion_t(sample_weights, out_t, self.num_pos)

        loss = s_loss + t_loss
        sample_loss = s_sample_loss + t_sample_loss
        return loss, sample_loss


class ContrastLoss(nn.Module):
    """
    取query对应类别的class centers作为正样本对
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, sample_weights, x, num_pos):
        P = num_pos
        bsz = x.shape[0]
        m = x.size(1) - P

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.narrow(1, 0, P)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()
        # print("positive:", log_D1.shape)

        # loss for K negative pair
        P_neg = x.narrow(1, P, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()
        # print("negative:", log_D0.shape)

        if P > 1:
            sample_loss = - ((log_D1.squeeze() + log_D0.sum(1).view(bsz, 1).repeat(1, P))).sum(1) / P
            # loss = (sample_weights.view(-1) * sample_loss).sum(0)/bsz
        elif P == 1:
            sample_loss = - (log_D1.squeeze() + log_D0.sum(1).squeeze())
            # loss = (sample_weights.view(-1) * sample_loss).sum(0)/bsz
        sample_loss = (sample_weights.view(-1) * sample_loss)
        loss = sample_loss.sum(0)/bsz

        return loss, sample_loss



class ContrastLoss_v2(nn.Module):
    """
    取KNN neighbors作为正样本对, 并且根据neighbors和query之间的相似度给pos pairs分配权重。
    """
    def __init__(self, n_data):
        super(ContrastLoss_v2, self).__init__()
        self.n_data = n_data

    def forward(self, sample_weights, x, num_pos, knn_similarity):
        P = num_pos
        bsz = x.shape[0]
        m = x.size(1) - P
        # print("knn_similarity:", knn_similarity)

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.narrow(1, 0, P)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, P, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()
        # print("negative:", log_D0.shape)

        sample_loss = - (torch.multiply((log_D1.squeeze() + log_D0.sum(1).view(bsz, 1).repeat(1, P)), \
            knn_similarity)).sum(1) / knn_similarity.sum(1)
        # print("sample_loss_v2:", sample_loss)

        sample_loss = (sample_weights.view(-1) * sample_loss)
        loss = sample_loss.sum(0)/bsz
        # print("CRD loss:", loss)

        return loss, sample_loss



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


