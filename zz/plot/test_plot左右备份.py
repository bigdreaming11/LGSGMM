import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

def plot_yinshe_zuoyou(match00, match_filt, match_new, image0, image1, boxes0, boxes1, output_folder, img_file_a,buchong_id, buchong_bbox):
    def tensor_to_npimg(tensor):
        img = tensor.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # (C, H, W) → (H, W, C)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def draw_boxes(img, boxes, color=(0, 255, 0)):
        for box in boxes:
            cx = int(box[2])
            cy = int(box[3])
            bw = int(box[4])
            bh = int(box[5])
            top_left = (cx - bw // 2, cy - bh // 2)
            bottom_right = (cx + bw // 2, cy + bh // 2)
            cv2.rectangle(img, top_left, bottom_right, color, 2)

    def get_centers(boxes):
        centers = []
        for box in boxes:
            cx = int(box[2])
            cy = int(box[3])
            centers.append((cx, cy))
        return centers

    def plot_matches(image0, image1, boxes0, boxes1, matches, color, ax):
        centers0 = get_centers(boxes0)
        centers1 = get_centers(boxes1)
        for i, j in matches:
            pt1 = centers0[i]
            pt2 = (centers1[j][0] + image0.shape[1], centers1[j][1])  # 右图中心坐标要加宽度偏移
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=0.8)
            ax.scatter(*pt1, color=color, s=6)
            ax.scatter(*pt2, color=color, s=6)

    def plot_unmatched_points_and_boxes(ax, img0, boxes1, buchong_id, buchong_bbox):
    # """
    # 在图A（左图，img0）上画红色框（映射未匹配成功的框），
    # 在图B（右图）上画红点（对应中心），并用红虚线连接两点。

    # 参数：
    # - ax: matplotlib的axes对象，用于画线和点
    # - img0: numpy图像（图A），会直接在其上用cv2画矩形框
    # - boxes1: 图B中的所有框，用来获取宽高
    # - buchong_id: 图B中未匹配的框的索引列表
    # - buchong_bbox: 图B映射到图A后的对应中心点坐标列表（[[x, y], ...]）
    # """
        centers1 = [(int(box[2]), int(box[3])) for box in boxes1]

        for idx_b, warped_center in zip(buchong_id, buchong_bbox):
            cx, cy = warped_center
            bw = int(boxes1[idx_b][4])
            bh = int(boxes1[idx_b][5])

            # 图A上画红框
            top_left = (int(cx - bw // 2), int(cy - bh // 2))
            bottom_right = (int(cx + bw // 2), int(cy + bh // 2))

            cv2.rectangle(img0, top_left, bottom_right, (255, 0, 0), 2) 
            # 图B上中心点 + 宽度偏移
            pt_b = centers1[idx_b]
            pt_b_shifted = (pt_b[0] + img0.shape[1], pt_b[1])

            # 画红色虚线连接
            ax.plot([cx, pt_b_shifted[0]], [cy, pt_b_shifted[1]], color='red',  linewidth=1)
            
            # 图B上红点
            ax.scatter(*pt_b_shifted, color='red', s=10, marker='o')




    # 确保输出路径存在
    os.makedirs(output_folder, exist_ok=True)

    # 文件名去掉扩展名
    filename_base = os.path.splitext(os.path.basename(img_file_a))[0]  # e.g. '00001'

    # -------- 图 1：filt --------
    img0 = tensor_to_npimg(image0).copy()
    img1 = tensor_to_npimg(image1).copy()

    draw_boxes(img0, boxes0)
    draw_boxes(img1, boxes1)

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    img_combined1 = np.concatenate([img0, img1], axis=1)
    ax1.imshow(img_combined1)
    ax1.axis("off")

    match00_set = set(map(tuple, match00))
    match_filt_set = set(map(tuple, match_filt))
    removed = list(match00_set - match_filt_set)

    plot_matches(img0, img1, boxes0, boxes1, match00, color='yellow', ax=ax1)
    plot_matches(img0, img1, boxes0, boxes1, removed, color='blue', ax=ax1)

    save_path1 = os.path.join(output_folder, f"{filename_base}_filt.png")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path1, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig1)

    # -------- 图 2：buchong --------
    img0 = tensor_to_npimg(image0).copy()
    img1 = tensor_to_npimg(image1).copy()
    draw_boxes(img0, boxes0)
    draw_boxes(img1, boxes1)


    fig2, ax2 = plt.subplots(figsize=(12, 6))
    plot_unmatched_points_and_boxes(ax2, img0, boxes1, buchong_id, buchong_bbox)
    img_combined2 = np.concatenate([img0, img1], axis=1)
    ax2.imshow(img_combined2)
    ax2.axis("off")

    match_new_set = set(map(tuple, match_new))
    new_add = list(match_new_set - match_filt_set)

    plot_matches(img0, img1, boxes0, boxes1, match_filt, color='yellow', ax=ax2)
    # plot_matches(img0, img1, boxes0, boxes1, new_add, color='yellow', ax=ax2)



    save_path2 = os.path.join(output_folder, f"{filename_base}_buchong.png")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path2, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig2)

    print(f"✅ 匹配图保存至：\n  {save_path1}\n  {save_path2}")
