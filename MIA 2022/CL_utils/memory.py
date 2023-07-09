import torch
import numpy as np
from torch import nn
import math


class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
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

        return out_v1, out_v2


class ContrastMemory_v2(nn.Module):
    """
    memory buffer that supplies large amount of positive and negative samples.
    """
    def __init__(self, inputSize, outputSize, P, K, T=0.07, momentum=0.5, select_pos_pairs=True, P2=10):
        super(ContrastMemory_v2, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.P = P
        self.K = K
        self.P2 = P2
        self.select_pos_pairs = select_pos_pairs

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum, P]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        "v1 is the feature of the student model, v2 refer to the teacher feature."

        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        P = int(self.params[5].item())

        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + P)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + P, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + P, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        """
        筛选对比学习中的正样本。分别计算两个网络的query batch和memory bank中所有正样本之间的relation matrix.
        如果网络1中similarity大于网络2的正样本。
        认为这些样本对于网络2的学习有难度，用这些正样本计算CRD loss进行网络1到网络2的蒸馏。
        """
        # print("sample index:", y)
        # print("positive pairs:", idx)
        # print("out v1:", out_v1.shape, out_v1)
        if self.select_pos_pairs == True:
            # print("weight_v1:", weight_v1.shape, "v1:", v1.shape)
            v1_relation = torch.bmm(weight_v1/torch.norm(weight_v1, dim=2, keepdim=True), \
                (v1/torch.norm(v1, dim=1, keepdim=True)).view(batchSize, inputSize, 1))
            # print("v1 positive:", torch.mean(v1_relation.narrow(1, 0, P)))
            # print("v1 negative:", torch.mean(v1_relation.narrow(1, P, K)))

            v2_relation = torch.bmm(weight_v2/torch.norm(weight_v2, dim=2, keepdim=True), \
                (v2/torch.norm(v2, dim=1, keepdim=True)).view(batchSize, inputSize, 1))

            v1_relation_pos, v2_relation_pos = v1_relation.narrow(1, 0, P), v2_relation.narrow(1, 0, P)
            ### 排序，对batch中的每个样本，找到在两个网络中relation差别最大的10个正样本。
            sorted, indices_v1 = torch.sort(v2_relation_pos-v1_relation_pos, dim=1, descending=True)
            # print(y)
            # print(indices_v1[:, 0, :])
            indices_v1[:, 0, :] = 0
            # print("all indices:", indices_v1.shape)

            # print("sample index:", y)
            selected_indices_v1 = indices_v1[:, :self.P2, :].squeeze()
            # print("selected indices:", selected_indices_v1)
            sample_index = torch.arange(0, out_v1.shape[0], 1).view(-1,1).repeat(1, self.P2).cuda()
            selected_indices_v1 = (sample_index * (K + P) + selected_indices_v1).view(-1)
            # print("selected indices:", selected_indices_v1.shape)

            # print("out_v1", out_v1.shape)
            out_v1_pos = out_v1.view(-1, 1).index_select(0, selected_indices_v1).view(-1, self.P2, 1)
            out_v1_neg = out_v1.narrow(1, P, K)
            # print("selected positives:", out_v1_pos.shape)
            # print("negatives:", out_v1_neg.shape)
            out_v1 = torch.cat((out_v1_pos, out_v1_neg), 1)
            # print(out_v1.shape)

            _, indices_v2 = torch.sort(v1_relation_pos-v2_relation_pos, dim=1, descending=True)
            indices_v2[:, 0, :] = 0
            selected_indices_v2 = indices_v2[:, :self.P2, :].squeeze()
            # print("selected indices:", selected_indices_v2)
            selected_indices_v2 = (sample_index * (K + P) + selected_indices_v2).view(-1)

            out_v2_pos = out_v2.view(-1, 1).index_select(0, selected_indices_v2).view(-1, 10, 1)
            out_v2_neg = out_v2.narrow(1, P, K)
            out_v2 = torch.cat((out_v2_pos, out_v2_neg), 1)

        # print("out v1 after selection:", out_v1.shape, out_v1)

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

        return out_v1, out_v2


class ContrastMemory_v3(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    Only construct memory buffer for the teacher model.
    """
    def __init__(self, inputSize, outputSize, P, K, T=0.07, momentum=0.5, select_pos_pairs=True, P2=10):
        super(ContrastMemory_v3, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.P = P
        self.K = K
        self.P2 = P2
        self.select_pos_pairs = select_pos_pairs

        self.register_buffer('params', torch.tensor([P, K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, epoch, v1, v2, y, idx=None, select_pos_mode="hard"):
        """
        update the teacher's memory buffer.
        :param v1: teacher feature. [bs, feat_dim]
        :param v2: student feature. [bs, feat_dim]
        :param y: the index of the samples in the mini-batch.
        :param idx: the contrastive samples' features of each batch sample. [bs, K+P, feat_dim]
        """
        P = int(self.params[0].item())
        K = int(self.params[1].item())
        T = self.params[2].item()
        Z_v2 = self.params[3].item()
        momentum = self.params[4].item()

        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + P)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + P, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        # print(out_v2.mean())
        out_v2 = torch.exp(torch.div(out_v2, T)) ## [bs, K+P, 1]

        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + P, inputSize)

        """
        筛选对比学习中的正样本。计算student和teacher中的query和teacher memory bank中所有正样本之间的relation matrix.
        如果teacher的pos. similarity 比学生高很多
        认为这些样本对于学生的学习有难度，用这些正样本计算CRD loss进行teacher到student的蒸馏。
        """
        # print("sample index:", y)
        # print("positive pairs:", idx)
        # print("out v1:", out_v1.shape, out_v1)
        if self.select_pos_pairs == True:
            # print("weight_v1:", weight_v1.shape, "v1:", v1.shape)
            t_relation = torch.bmm(weight_v1/torch.norm(weight_v1, dim=2, keepdim=True), \
                (v1/torch.norm(v1, dim=1, keepdim=True)).view(batchSize, inputSize, 1))

            s_relation = torch.bmm(weight_v2/torch.norm(weight_v2, dim=2, keepdim=True), \
                (v2/torch.norm(v2, dim=1, keepdim=True)).view(batchSize, inputSize, 1))

            t_relation_pos, s_relation_pos = t_relation.narrow(1, 0, P), s_relation.narrow(1, 0, P)

            ### 排序，对batch中的每个样本，找到在两个网络中relation差别最大的10个正样本。
            # print("relation difference:", torch.mean(t_relation_pos-s_relation_pos))
            indices = torch.sort(t_relation_pos-s_relation_pos, dim=1, descending=True)[1]
            # print("all indices:", indices_v1.shape)

            # print("sample index:", y)
            if select_pos_mode == "hard":
                selected_indices = indices[:, :self.P2, :].squeeze(-1) # 取差别最大的10个正样本
            elif select_pos_mode == "mid":
                index = torch.tensor(np.random.randint(50, 100, self.P2)).cuda()
                # print("index:", index)
                selected_indices = indices.index_select(1, index).squeeze(-1)
            elif select_pos_mode == "random":
                index = torch.tensor(np.random.randint(0, self.P, self.P2)).cuda()
                selected_indices = indices.index_select(1, index).squeeze(-1)
            elif select_pos_mode == "curriculum":
                interval = 4 - np.ceil(3*epoch)
                index = torch.tensor(np.random.randint(50*(interval-1), 50*interval, self.P2)).cuda()
                selected_indices = indices.index_select(1, index).squeeze(-1)
            
            # print("selected_indices:", selected_indices)
            selected_indices[:, 0] = 0
            # print("selected_indices:", selected_indices)
            # print("selected indices:", idx[0].index_select(0, selected_indices[0].view(-1)))
            sample_index = torch.arange(0, out_v2.shape[0], 1).view(-1,1).repeat(1, self.P2).cuda()
            selected_indices = (sample_index * (K + P) + selected_indices).view(-1)
            # print("selected indices:", selected_indices_v1.shape)

            out_v2_pos = out_v2.view(-1, 1).index_select(0, selected_indices).view(-1, self.P2, 1)
            out_v2_neg = out_v2.narrow(1, P, K)
            out_v2 = torch.cat((out_v2_pos, out_v2_neg), 1)


        # set Z if haven't been set yet
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v2, the similarity between student samples and (K+P) teacher anchors.
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

        return out_v2, self.memory_v1


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj