import time

from lightglue.lightglue import LightGlue,LightGlue_reid


import torch
import torch.nn as nn
from zz.train.util import move_to_cuda
from zz.all_nn.backbone import ResNetFeatureExtractor
from zz.all_nn.roi_pooling import  CustomRoIPooling


class TargetConnectionModel(nn.Module):
    def __init__(self,backbone_out_dim=None,filter_threshold=None,lightglue_pretrain=None):
        super(TargetConnectionModel, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ##backbone
        self.backbone=ResNetFeatureExtractor().to(device)
        ##roi
        self.roi=CustomRoIPooling(output_dim=backbone_out_dim).to(device)
        ##superglue
        self.lightgflue=LightGlue(backbone_out_dim=backbone_out_dim,filter_threshold=filter_threshold,lightglue_pretrain=lightglue_pretrain)

    def forward(self, images_a, images_b, keypoints_a, keypoints_b, mask_a, mask_b):
        # 提取特征\
        st=time.time()
        features_map_a = self.backbone(images_a)
        features_map_b = self.backbone(images_b)
        end=time.time()
        # print(f"分开算法时间为{end-st}s")


        original_H=images_a.shape[2]
        original_W=images_a.shape[3]
        features_a=self.roi(features_map_a,keypoints_a,mask_a, original_H, original_W)
        features_b=self.roi(features_map_b,keypoints_b,mask_b, original_H, original_W)

        feature_patch_a_tensor = {
            "descriptors": features_a,
            "keypoints":  keypoints_a[:, :, :2].float(),
            "image_size": torch.tensor([original_W, original_H], dtype=torch.int32).unsqueeze(0),
            "mask": mask_a
        }

        feature_patch_b_tensor = {
            "descriptors": features_b,
            "keypoints":  keypoints_b[:, :, :2].float(),
            "image_size": torch.tensor([original_W, original_H], dtype=torch.int32).unsqueeze(0),
            "mask": mask_b
        }

        feats0 = move_to_cuda(feature_patch_a_tensor)
        feats1 = move_to_cuda(feature_patch_b_tensor)

        # 匹配
        matcher = self.lightgflue
        matches01 = matcher({'image0': feats0, 'image1': feats1})

        matches = matches01['matches']

        points0 = feats0['keypoints']
        points1 = feats1['keypoints']
        scores = matches01['scores']

        self_attn0 = matches01['self_attn0']
        self_attn1 = matches01['self_attn1']        
        cross_attn01 = matches01['cross_attn01']
        cross_attn10 = matches01['cross_attn10']     

        return points0, points1, matches, scores, features_a, features_b,self_attn0,self_attn1,cross_attn01,cross_attn10
    
class TargetConnectionModel_reid(nn.Module):
    def __init__(self,backbone_out_dim=None,filter_threshold=None,lightglue_pretrain=None):
        super(TargetConnectionModel_reid, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ##backbone
        self.backbone=ResNetFeatureExtractor().to(device)
        ##roi
        self.roi=CustomRoIPooling(output_dim=backbone_out_dim).to(device)
        ##superglue
        self.lightgflue=LightGlue_reid(backbone_out_dim=backbone_out_dim,filter_threshold=filter_threshold,lightglue_pretrain=lightglue_pretrain)

    def forward(self, images_a, images_b, keypoints_a, keypoints_b, mask_a, mask_b):
        # 提取特征\
        st=time.time()
        features_map_a = self.backbone(images_a)
        features_map_b = self.backbone(images_b)
        end=time.time()
        # print(f"分开算法时间为{end-st}s")


        original_H=images_a.shape[2]
        original_W=images_a.shape[3]
        features_a=self.roi(features_map_a,keypoints_a,mask_a, original_H, original_W)
        features_b=self.roi(features_map_b,keypoints_b,mask_b, original_H, original_W)

        feature_patch_a_tensor = {
            "descriptors": features_a,
            "keypoints":  keypoints_a[:, :, :2].float(),
            "image_size": torch.tensor([original_W, original_H], dtype=torch.int32).unsqueeze(0),
            "mask": mask_a
        }

        feature_patch_b_tensor = {
            "descriptors": features_b,
            "keypoints":  keypoints_b[:, :, :2].float(),
            "image_size": torch.tensor([original_W, original_H], dtype=torch.int32).unsqueeze(0),
            "mask": mask_b
        }

        feats0 = move_to_cuda(feature_patch_a_tensor)
        feats1 = move_to_cuda(feature_patch_b_tensor)

        # 匹配
        matcher = self.lightgflue
        matches01 = matcher({'image0': feats0, 'image1': feats1})

        matches = matches01['matches']

        points0 = feats0['keypoints']
        points1 = feats1['keypoints']
        scores = matches01['scores']

        self_attn0 = matches01['self_attn0']
        self_attn1 = matches01['self_attn1']        
        cross_attn01 = matches01['cross_attn01']
        cross_attn10 = matches01['cross_attn10']     

        return points0, points1, matches, scores, features_a, features_b,self_attn0,self_attn1,cross_attn01,cross_attn10