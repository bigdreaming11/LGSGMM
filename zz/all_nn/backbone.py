
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import torch
import torchvision.models as models
from torch import nn
import torch
import torch.nn as nn
import numpy as np



import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
from collections import OrderedDict
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone=models.resnet50(pretrained=True)):
        super(ResNetFeatureExtractor, self).__init__()
        # 只保留 ResNet 的前 4 层
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

    def forward(self, x):
        # 提取第 4 层的完整特征图
        return self.backbone(x)
    
########################
####OSNET##############
######################
# class ResNetFeatureExtractor(nn.Module):
#     def __init__(self, model_name='osnet_x1_0',weight_path=r"/mnt/8t/zz/zz/baseline_superglue_TMM/weights/osnet_x1_0_imagenet.pth"):
#         super().__init__()
#         self.model = build_model(
#             name=model_name,
#             num_classes=1000,
#             loss='softmax',
#             pretrained=False
#         )

#         if weight_path is not None:
#             checkpoint = torch.load(weight_path, map_location='cpu')
#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             else:
#                 state_dict = checkpoint
#             new_state_dict = OrderedDict()
#             for k, v in state_dict.items():
#                 name = k.replace('module.', '')
#                 new_state_dict[name] = v
#             self.model.load_state_dict(new_state_dict, strict=False)

#     def forward(self, x):
#         # 使用 model.featuremaps(x) 获取 backbone 特征图
#         return self.model.featuremaps(x)  # 输出 [B, 512, Hf, Wf]



#################################################################### AGW的在resnet50基础上改进￥￥￥￥￥￥￥￥￥￥￥￥￥￥
# import torch
# import torch.nn as nn
# from torchvision import models

# class GeneralizedMeanPoolingP(nn.Module):
#     """GeM pooling with learnable p"""
#     def __init__(self, norm=3.0, eps=1e-6):
#         super(GeneralizedMeanPoolingP, self).__init__()
#         self.p = nn.Parameter(torch.ones(1) * norm)
#         self.eps = eps

#     def forward(self, x):
#         # x: [B, C, H, W]
#         x = torch.clamp(x, min=self.eps)
#         x = x.pow(self.p)
#         x = x.mean(dim=(-2, -1))
#         x = x.pow(1.0 / self.p)
#         return x.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

# class ResNetFeatureExtractor(nn.Module):
#     """
#     ResNet50 backbone + GeM pooling + BN，只输出特征
#     """
#     def __init__(self, pretrained=True):
#         super(ResNetFeatureExtractor, self).__init__()
#         backbone = models.resnet50(pretrained=pretrained)

#         # backbone: 只保留到 layer4
#         self.backbone = nn.Sequential(
#             backbone.conv1,
#             backbone.bn1,
#             backbone.relu,
#             backbone.maxpool,
#             backbone.layer1,
#             backbone.layer2,
#             backbone.layer3,
#             backbone.layer4
#         )
#         self.out_channels = 2048

#         # GeM 池化
#         self.global_pool = GeneralizedMeanPoolingP()

#         # BN
#         self.bottleneck = nn.BatchNorm1d(self.out_channels)
#         self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.apply(self._weights_init_kaiming)

#     def _weights_init_kaiming(self, m):
#         classname = m.__class__.__name__
#         if classname.find('BatchNorm') != -1 and m.affine:
#             nn.init.constant_(m.weight, 1.0)
#             nn.init.constant_(m.bias, 0.0)

#     def forward(self, x):
#         feat_map = self.backbone(x)             # [B, 2048, H/32, W/32]
#         global_feat = self.global_pool(feat_map)  # [B, 2048, 1, 1]
#         global_feat = global_feat.view(global_feat.size(0), -1)  # [B, 2048]
#         feat = self.bottleneck(global_feat)     # BN 处理
#         return feat_map                  # 输出特征图 + 全局特征
