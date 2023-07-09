"""
Paper: Knowledge distillation from multi-modal to mono-modal segmentation networks.
除了对于prediction做KL div蒸馏之外，还对中间feature embeddings用KL div蒸馏。
"""

from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class feats_KL(nn.Module):
    def __init__(self):
        super(feats_KL, self).__init__()

    def forward(self, f_s, f_t):
        f_s = F.log_softmax(f_s, dim=1)
        f_t = F.softmax(f_t, dim=1)
        loss = F.kl_div(f_s, f_t, size_average=False) / f_s.shape[0]
        return loss
