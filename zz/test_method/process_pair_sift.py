
import torch
from lightglue.utils import load_image, rbd
from zz.dataloader.txt_img_load import load_txt
from zz.dataloader.dataset import pad_and_mask
import cv2
import os
from duibi_ceshi.zz_sift import zz_sift 
def process_pair_sift( img_path_a, txt_path_a, img_path_b, txt_path_b, output_path):
    """
    处理一对图片及其对应的 txt 文件
    :param img_path_a: 图片 A 的路径
    :param txt_path_a: 文本文件 A 的路径
    :param img_path_b: 图片 B 的路径
    :param txt_path_b: 文本文件 B 的路径
    :param output_path: 输出结果保存路径
    """
    # 加载图片和 txt 文件
    image0, img0_np, img0_width, img0_height = load_image(img_path_a)
    image1, img1_np, img1_width, img1_height = load_image(img_path_b)
    boxes0 = load_txt(txt_path_a, img0_width, img0_height)
    boxes1 = load_txt(txt_path_b, img1_width, img1_height)

    ##########################################sift算法###############################################################
    matches, H, index=zz_sift(image0,image1,boxes0,boxes1)               
    # matches, H, index=zz_topu(img0_np,img1_np,boxes0,boxes1)        
    return boxes0 ,boxes1,image0,image1,matches,H,index