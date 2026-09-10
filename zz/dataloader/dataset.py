import os
import glob
import torch
from torch.utils.data import Dataset
from lightglue.utils import load_image
from zz.dataloader.txt_img_load import load_txt
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.nn.parallel import DataParallel
import torch.nn as nn


class TargetConnectionDataset(Dataset):

    def __init__(self, train_val_path):
        self.txt_a = train_val_path[2]
        self.txt_b = train_val_path[3]
        self.img_a = train_val_path[0]
        self.img_b = train_val_path[1]

        # 验证路径有效性
        for path in train_val_path:
            assert os.path.exists(path), f"Path does not exist: {path}"

        # 加载图像文件列表
        self.image_files_a = self._get_image_files(self.img_a)
        self.image_files_b = self._get_image_files(self.img_b)

    def _get_image_files(self, directory):
        supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for ext in supported_extensions:
            files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
        return sorted(files)

    def __len__(self):
        return min(len(self.image_files_a), len(self.image_files_b))

    def __getitem__(self, idx):
        # 根据索引获取图像文件路径
        img_a_path = self.image_files_a[idx]
        img_b_path = self.image_files_b[idx]

        # 确保文件对齐
        assert os.path.basename(img_a_path).split('.')[0] == os.path.basename(img_b_path).split('.')[0], \
            f"File mismatch: {img_a_path} and {img_b_path}"
        # 加载图像
        image0, img0_np, img0_width, img0_height = load_image(img_a_path)#image0是tensor
        image1, img1_np, img1_width, img1_height = load_image(img_b_path)
        # 加载标注文件
        txt_a_path = os.path.join(self.txt_a, os.path.basename(img_a_path).split('.')[0] + '.txt')
        txt_b_path = os.path.join(self.txt_b, os.path.basename(img_b_path).split('.')[0] + '.txt')
        boxes0 = load_txt(txt_a_path, img0_width, img0_height)
        boxes1 = load_txt(txt_b_path, img1_width, img1_height)
        # 提取目标区域
        patch_a = []
        patch_b = []
        for box in boxes0:
            class_id, id, x, y, w, h = box[0], box[1], box[2], box[3], box[4], box[5]
            patch_a.append([ class_id, id,x,y,w,h])
        for box in boxes1:
            class_id, id, x, y, w, h = box[0], box[1], box[2], box[3], box[4], box[5]
            patch_b.append([class_id, id,x,y,w,h])


        patches_a =image0
        class_id_a = [[p[0], p[1]] for p in patch_a]#注意这里的keypoints_a就是class和id了
        keypoints_a=[[p[2], p[3],p[4],p[5]] for p in patch_a]
        patches_b =  image1
        class_id_b = [[p[0], p[1]] for p in patch_b]
        keypoints_b=[[p[2], p[3],p[4],p[5]]  for p in patch_b]
        return {
            "image_a": image0,
            "image_b": image1,
            'boxes_a': class_id_a,
            'boxes_b': class_id_b,
            "keypoints_a":keypoints_a,
            "keypoints_b":keypoints_b,
        }

def custom_collate_fn(batch):
    """
    自定义 collate_fn，将批次数据合并成一个字典。
    dd=0
    """
    dd=0
    return {
        "image_a":  torch.stack([item["image_a"] for item in batch]),  # 列表，每个元素是一个图像块列表
        "image_b":  torch.stack([item["image_b"] for item in batch]),  # 列表，每个元素是一个图像块列表
        "boxes_a": [item["boxes_a"] for item in batch],  # 列表，每个元素是一个关键点坐标列表
        "boxes_b": [item["boxes_b"] for item in batch],  # 列表，每个元素是一个关键点坐标列表
        "keypoints_a": [item["keypoints_a"] for item in batch],  # 列表，每个元素是一个关键点坐标列表
        "keypoints_b": [item["keypoints_b"] for item in batch],  # 列表，每个元素是一个关键点坐标列表
    }# 直接返回输入的批次列表




#################################################################################
def pad_and_mask(patches, max_len):
    """
    对 patches 进行 padding 并生成 mask。
    :param patches: List[List[patch]]，每个 patch 是一个小图像切片。
    :param max_len: 最大切片数量。
    :return: padded_patches (numpy array), mask (bool tensor)
    """
    padded_patches = []
    mask = []

    for patch_list in patches:
        # 计算当前样本的切片数量
        current_len = len(patch_list)

        # Padding 到 max_len
        padded_patches.append(patch_list + [np.zeros_like(patch_list[0])] * (max_len - current_len))

        # 生成 mask
        mask.append([True] * current_len + [False] * (max_len - current_len))

    return torch.tensor(padded_patches), torch.tensor(mask, dtype=torch.bool)

# def pad_and_mask(patches, max_len):
#     """
#     对 patches 进行 padding 并生成 mask。
#     :param patches: List[List[patch]]，每个 patch 是一个小图像切片（list 或 array）。
#     :param max_len: 最大切片数量。
#     :return: padded_patches (tensor), mask (bool tensor)
#     """
#     padded_patches = []
#     mask = []

#     # 找一个非空样本推断形状
#     example_patch = None
#     for patch_list in patches:
#         if len(patch_list) > 0:
#             example_patch = np.zeros_like(patch_list[0])
#             break
#     if example_patch is None:
#         raise ValueError("所有 patch_list 都是空的，无法推断 patch 的形状。")

#     for patch_list in patches:
#         current_len = len(patch_list)

#         if current_len == 0:
#             # 空的就直接补全
#             padded = [np.zeros_like(example_patch)] * max_len
#             mask.append([False] * max_len)
#         else:
#             # 根据实际 patch 的 shape 来补
#             padded = patch_list + [np.zeros_like(patch_list[0])] * (max_len - current_len)
#             mask.append([True] * current_len + [False] * (max_len - current_len))

#         padded_patches.append(padded)

#     return torch.tensor(np.array(padded_patches)), torch.tensor(mask, dtype=torch.bool)



class CustomDataParallel(nn.DataParallel):
    def gather(self, outputs, output_device):
        """
        自定义 gather 方法，支持张量和列表的合并。
        - 对于张量：使用 torch.cat 按批次叠加。
        - 对于列表：使用 extend 合并为一个大的列表。
        """
        # 初始化结果容器
        gathered_outputs = []

        # 假设每个 GPU 的输出是一个元组 (output1, output2, ...)
        for i in range(len(outputs[0])):  # 遍历每个字段
            field_list = [output[i] for output in outputs]  # 提取该字段的所有 GPU 输出

            if isinstance(field_list[0], torch.Tensor):  # 如果是张量
                # 将张量移动到主设备
                field_list = [tensor.to(output_device) for tensor in field_list]
                # 使用 torch.cat 按批次叠加
                gathered_field = torch.cat(field_list, dim=0)
            elif isinstance(field_list[0], list):  # 如果是列表
                # 使用 extend 合并为一个大的列表
                gathered_field = []
                for field in field_list:
                    gathered_field.extend(field)
            else:
                raise TypeError(f"Unsupported type: {type(field_list[0])}. Only tensors and lists are supported.")

            # 添加到结果容器
            gathered_outputs.append(gathered_field)

        # 返回结果
        return tuple(gathered_outputs)

