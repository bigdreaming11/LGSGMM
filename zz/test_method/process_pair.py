import torch
from lightglue.utils import load_image, rbd
from zz.dataloader.txt_img_load import load_txt
from zz.dataloader.dataset import pad_and_mask
import cv2
import os
import time
def draw_and_save(img_np, boxes, save_path, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制目标框并保存
    :param img_np: numpy图像 (H,W,3)
    :param boxes: list，每个元素是 [cls, conf, x, y, w, h]
    :param save_path: 保存路径
    :param color: 框的颜色 (B,G,R)
    :param thickness: 框线条粗细
    """
    img = img_np.copy()

    for box in boxes:
        cls, conf, x, y, w, h = box
        # 转换成左上角和右下角坐标
        x1, y1 = int(x-0.5*(w)), int(y-0.5*(h))
        x2, y2 = int(x + 0.5*w), int(y +0.5* h)


        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 在框上写类别和置信度
        label = f"{cls}:{conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 创建保存文件夹
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"✅ 结果已保存: {save_path}")




def process_pair(device,model, img_path_a, txt_path_a, img_path_b, txt_path_b):
    """
    处理一对图片及其对应的 txt 文件
    :param img_path_a: 图片 A 的路径
    :param txt_path_a: 文本文件 A 的路径
    :param img_path_b: 图片 B 的路径
    :param txt_path_b: 文本文件 B 的路径
    :param output_path: 输出结果保存路径
    """
    # 加载图片和 txt 文件
    start_time=time.time()
    image0, img0_np, img0_width, img0_height = load_image(img_path_a)
    end_time=time.time()
    print(f"算法的运推理时间：{(end_time-start_time)*1000}ms")
    image1, img1_np, img1_width, img1_height = load_image(img_path_b)


    boxes0 = load_txt(txt_path_a, img0_width, img0_height)
    boxes1 = load_txt(txt_path_b, img1_width, img1_height)

        # 示例：保存两个图像的可视化
    # draw_and_save(img0_np, boxes0, "/mnt/8t/zz/zz/baseline_superglue_TMM/out_match/00/img1_with_boxes.jpg", color=(0, 255, 0))
    # draw_and_save(img1_np, boxes1, "/mnt/8t/zz/zz/baseline_superglue_TMM/out_match/00/img2_with_boxes.jpg", color=(255, 0, 0))
    # 提取目标区域
    patch_a = [[box[0], box[1], box[2], box[3], box[4], box[5]] for box in boxes0]
    patch_b = [[box[0], box[1], box[2], box[3], box[4], box[5]] for box in boxes1]

    keypoints_a = [[[p[2], p[3], p[4], p[5]] for p in patch_a]]
    class_id_a = [[[p[0], p[1]] for p in patch_a]]
    keypoints_b = [[[p[2], p[3], p[4], p[5]] for p in patch_b]]
    class_id_b = [[[p[0], p[1]] for p in patch_b]]

    # Padding 和 Masking
    max_len_a = max(len(patches) for patches in class_id_a)
    max_len_b = max(len(patches) for patches in class_id_b)

    padded_keypoints_a, mask_a = pad_and_mask(keypoints_a, max_len_a)
    padded_boxes_a, _ = pad_and_mask(class_id_a, max_len_a)
    padded_keypoints_b, mask_b = pad_and_mask(keypoints_b, max_len_b)
    padded_boxes_b, _ = pad_and_mask(class_id_b, max_len_b)

    # 将所有张量移动到设备

    images_a = image0.to(device).unsqueeze(0)
    images_b = image1.to(device).unsqueeze(0)


    padded_keypoints_a = padded_keypoints_a.to(device)
    mask_a = mask_a.to(device)
    padded_boxes_a = padded_boxes_a.to(device)
    padded_keypoints_b = padded_keypoints_b.to(device)
    mask_b = mask_b.to(device)
    padded_boxes_b = padded_boxes_b.to(device)



    # # 模型前向传播
    # with torch.no_grad():
    #     points0, points1, matches, scores, features_a, features_b ,self_attn0,self_attn1,cross_attn01,cross_attn10,lap_pe0,lap_pe1= self.model(
    #         images_a, images_b, padded_keypoints_a, padded_keypoints_b, mask_a, mask_b
    #     )
    with torch.no_grad():
        points0, points1, matches, scores, features_a, features_b ,self_attn0,self_attn1,cross_attn01,cross_attn10= model(
            images_a, images_b, padded_keypoints_a, padded_keypoints_b, mask_a, mask_b
        )

    matches_idx = matches[0][:]
    points00 = points0.squeeze(0)
    points11 = points1.squeeze(0)
    f0 = points00[matches_idx[:, 0]]
    f1 = points11[matches_idx[:, 1]]
######################################################################################
    matches00=matches[0]
    # return boxes0 ,boxes1,matches, features_a, features_b ,image0,image1,scores,self_attn0,self_attn1,cross_attn01,cross_attn10,lap_pe0,lap_pe1
    return boxes0 ,boxes1,matches, features_a, features_b ,image0,image1,scores,self_attn0,self_attn1,cross_attn01,cross_attn10
