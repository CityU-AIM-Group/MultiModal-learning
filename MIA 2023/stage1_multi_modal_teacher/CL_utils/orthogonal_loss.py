"""
Author: Xing Xiaohan
Date: 2022/03/15
This script constrain the multi-modal features to be orthogonal.
The source code is from https://github.com/fungtion/DSN/blob/master/functions.py.
"""

import torch
import torch.nn as nn

class OrthLoss(nn.Module):
    """
    Constraint the orthogonality of the features from different modalities.
    """

    def __init__(self):
        super(OrthLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss