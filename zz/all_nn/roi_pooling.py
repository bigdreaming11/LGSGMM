import torch.nn.functional as F
import torch
import torchvision.models as models
from torch import nn
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

class CustomRoIPooling(nn.Module):
    def __init__(self, output_dim=None):
        super(CustomRoIPooling, self).__init__()
        self.output_dim = output_dim

    def forward(self, feature_map, keypoints, mask, original_H, original_W):
        batch_size, max_boxes, _ = keypoints.shape
        channels, feature_H, feature_W = feature_map.shape[1], feature_map.shape[2], feature_map.shape[3]
        pooled_features = []

        # 计算下采样比例
        scale_x = feature_W / original_W
        scale_y = feature_H / original_H

        for b in range(batch_size):  # 遍历每个样本
            sample_pooled_features = []
            for i in range(max_boxes):  # 遍历每个目标框
                if mask[b, i] == 0:  # 如果掩码为 0，则跳过填充的目标框
                    sample_pooled_features.append(torch.zeros(self.output_dim).to(feature_map.device))
                    continue
                # 获取目标框的坐标
                x, y, w, h = keypoints[b, i]
                # 将目标框坐标映射到特征图空间
                x_resized = int(x * scale_x)
                y_resized = int(y * scale_y)
                w_resized = max(int(w * scale_x), 1)  # 确保宽度至少为 1
                h_resized = max(int(h * scale_y), 1)  # 确保高度至少为 1

                # 裁剪边界，确保目标框在特征图范围内
                x_resized = max(0, min(x_resized, feature_W - 1))
                y_resized = max(0, min(y_resized, feature_H - 1))
                w_resized = min(w_resized, feature_W - x_resized)
                h_resized = min(h_resized, feature_H - y_resized)

                # 从特征图中裁剪目标框区域
                roi_feature = feature_map[b, :, y_resized:y_resized + h_resized, x_resized:x_resized + w_resized]

                pooled = F.adaptive_avg_pool2d(roi_feature, (1, 1))  # [B, 2048, 1, 1]
                vec = pooled.contiguous().view(-1)  # 变成 [2048] 向量
                sample_pooled_features.append(vec)

            # 将所有目标框的特征堆叠成一个张量
            pooled_features.append(torch.stack(sample_pooled_features))

        # 将所有样本的特征堆叠成一个批次张量
        return torch.stack(pooled_features)