import numpy as np
import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from torch.nn import init, Parameter

# from my_utils.compute_gradients import get_grad_embedding

__all__ = ['ResNet', 'ResNet18', 'ResNet34', 'ResNet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

resnet_pretrained = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, path_dim=32, act=None, num_classes=7, return_grad = "False", zero_init_residual=False, 
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc_new = nn.Linear(512 * block.expansion, num_classes)
        # self.fc_new1 = nn.Sequential(nn.Linear(512 * block.expansion, 128),
        #                              nn.ReLU(inplace=True),
        #                              nn.Linear(128, path_dim),
        #                              nn.BatchNorm1d(path_dim),
        #                              nn.ReLU(inplace=True))
        self.fc_new1 = nn.Sequential(nn.Linear(512 * block.expansion, path_dim),
                                     nn.BatchNorm1d(path_dim),
                                     nn.ReLU(inplace=True))        
        # self.fc_new1 = nn.Linear(512 * block.expansion, 128)
        self.fc_new2 = nn.Linear(path_dim, num_classes)

        self.act = act
        self.return_grad = return_grad
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        fmaps_b1 = self.layer1(x)
        # print("layer1 output:", fmaps_b1.shape) # [bs, 64, 128, 128]
        fmaps_b2 = self.layer2(fmaps_b1)
        # print("layer2 output:", fmaps_b2.shape) # [bs, 128, 64, 64]
        fmaps_b3 = self.layer3(fmaps_b2)
        # print("layer3 output:", fmaps_b3.shape) # [bs, 256, 32, 32]
        fmaps_b4 = self.layer4(fmaps_b3)
        # print("layer4 output:", fmaps_b4.shape) # [bs, 512, 16, 16]


        feat_f3 = torch.flatten(self.avgpool(fmaps_b3), 1)
        x = self.avgpool(fmaps_b4)
        x = torch.flatten(x, 1)
        # fc_feat = x
        # print("feature:", x.shape) # [bs, 512]
        features = self.fc_new1(x)
        hazard = self.fc_new2(features)

        # if self.return_grad == "True":
        #     path_grads = get_grad_embedding(hazard, features).detach().cpu().numpy()
        # else:
        #     path_grads = None

        path_grads = None

        if self.act is not None:
            pred = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                pred = pred * self.output_range + self.output_shift

        # if self.return_grad == "True":
        #     y_c = torch.sum(hazard)
        #     features.grad = None
        #     features.retain_grad()
        #     y_c.backward(retain_graph=True)
        #     path_grads = features.grad.detach().cpu().numpy()
        #     path_grad_norm = np.linalg.norm(path_grads, axis=1)
        #     # print("gradient magnitude of the path feature:", path_grad_norm)
        #     # print("predicted hazard of the path branch:", np.reshape(hazard.detach().cpu().numpy(), (-1)))
        # else:
        #     path_grads = None

        return feat_f3, features, hazard, pred, path_grads


    def forward(self, **kwargs):
        x = kwargs['x_path']
        return self._forward_impl(x)


def _resnet(arch, block, layers, path_dim, act, num_classes, pretrained, progress, **kwargs):
    model = ResNet(block, layers, path_dim, act, num_classes, **kwargs)

    if pretrained:
        print("Loading pretrained model:", arch)
        # print('/home/meiluzhu2/data/pre_model/' + resnet_pretrained[arch])
        # checkpoint = torch.load('/home/meiluzhu2/data/pre_model/' + resnet_pretrained[arch])
        print('../pathomic_fusion_20221023_miccai/pretrained_resnet/' + resnet_pretrained[arch])
        checkpoint = torch.load('../pathomic_fusion_20221023_miccai/pretrained_resnet/' + resnet_pretrained[arch])        
        model.load_state_dict(checkpoint, strict=False)

    return model


def ResNet18(pretrained=True, progress=True, path_dim=32, act=None, num_classes=1, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], path_dim, act, num_classes, pretrained, progress,
                   **kwargs)


def ResNet34(pretrained=True, progress=True, num_classes=7, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], num_classes, pretrained, progress,
                   **kwargs)


def ResNet50(pretrained=True, progress=True, num_classes=7, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], num_classes, pretrained, progress,
                   **kwargs)
