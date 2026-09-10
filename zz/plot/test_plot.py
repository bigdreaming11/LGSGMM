import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def panduan(matches,boxes0,boxes1):
    true_match_idx=[]
    for num,match in enumerate(matches):
        idx1=match[0]
        idx2=match[1]
        if boxes0[idx1][0]==boxes1[idx2][0] and boxes0[idx1][1]==boxes1[idx2][1]:
            true_match_idx.append(num)    
    return true_match_idx

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

####################################
 ####做消融实验的时候，第一步不用过滤#
 ##################################
# def plot_matches(image0, image1, boxes0, boxes1, matches,removed_idx,true_match_idx, ax):
#     centers0 = get_centers(boxes0)
#     centers1 = get_centers(boxes1)

#     for k, (i, j) in enumerate(matches):
#         pt1 = centers0[i]
#         pt2 = (centers1[j][0] + image0.shape[1], centers1[j][1])

#         if k in true_match_idx:  # ✅ 正确匹配
#             color = 'yellow'
#             ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=0.8)
#             ax.scatter(*pt1, color=color, s=6, zorder=2)
#             ax.scatter(*pt2, color=color, s=6, zorder=2)

#         else:  # ❌ 错误匹配
#             ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='red', linewidth=0.8)
#             ax.scatter(*pt1, color=color, s=6, zorder=2)
#             ax.scatter(*pt2, color=color, s=6, zorder=2)
#             ax.scatter(*pt1, color='blue', marker='x', s=40, linewidths=1.5, zorder=999)
#             ax.scatter(*pt2, color='blue', marker='x', s=40, linewidths=1.5, zorder=999)
        # 在中点画蓝色 ×
        # ax.scatter(mid_x, mid_y, color='blue', marker='x', s=20, linewidths=1.2,zorder=999)
def plot_matches(image0, image1, boxes0, boxes1, matches, removed_idx, true_match_idx, ax):
    centers0 = get_centers(boxes0)
    centers1 = get_centers(boxes1)

    for k, (i, j) in enumerate(matches):
        pt1 = centers0[i]
        pt2 = (centers1[j][0] + image0.shape[1], centers1[j][1])

        # 线的颜色
        if k in true_match_idx:
            line_color = 'yellow'
        else:
            line_color = 'red'
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=line_color, linewidth=1, zorder=1)

        # # 散点颜色
        # if k in removed_idx:
        #     scatter_color = 'blue'
        #     marker = 'x'
        #     size = 40
        #     lw = 1.5
        # elif k in true_match_idx:
        #     scatter_color = 'yellow'
        #     marker = 'o'
        #     size = 6
        #     lw = None
        # else:
        #     scatter_color = 'red'
        #     marker = 'o'
        #     size = 6
        #     lw = None

                # 散点颜色

        if k in true_match_idx:
            scatter_color = 'yellow'
            marker = 'o'
            size = 6
            lw = None
        else:
            scatter_color = 'red'
            marker = 'o'
            size = 6
            lw = None
        
        ax.scatter(*pt1, color=scatter_color, marker=marker, s=size, linewidths=lw, zorder=2)
        ax.scatter(*pt2, color=scatter_color, marker=marker, s=size, linewidths=lw, zorder=2)

def plot_matches_2(image0, image1, boxes0, boxes1, matches, true_match_idx, ax):
    centers0 = get_centers(boxes0)
    centers1 = get_centers(boxes1)

    for k, (i, j) in enumerate(matches):
        pt1 = centers0[i]
        pt2 = (centers1[j][0] + image0.shape[1], centers1[j][1])

        if k in true_match_idx:  # ✅ 正确匹配
            color = 'yellow'
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=1)
            ax.scatter(*pt1, color=color, s=6, zorder=2)
            ax.scatter(*pt2, color=color, s=6, zorder=2)

        else:  # ❌ 错误匹配
            color = 'red'
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='red', linewidth=1)
            ax.scatter(*pt1, color=color, s=6, zorder=2)
            ax.scatter(*pt2, color=color, s=6, zorder=2)

        # 在中点画蓝色 ×
        # ax.scatter(mid_x, mid_y, color='blue', marker='x', s=20, linewidths=1.2,zorder=999)    


def plot_unmatched_points_and_boxes(ax, img0, boxes1, buchong_id, buchong_bbox):

    centers1 = [(int(box[2]), int(box[3])) for box in boxes1]

    for idx_b, warped_center in zip(buchong_id, buchong_bbox):
        cx, cy = warped_center
        bw = int(boxes1[idx_b][4])
        bh = int(boxes1[idx_b][5])

        # 图A上画红框
        top_left = (int(cx - bw // 2), int(cy - bh // 2))
        bottom_right = (int(cx + bw // 2), int(cy + bh // 2))

        # cv2.rectangle(img0, top_left, bottom_right, (255, 0, 0), 2) 
        rect = plt.Rectangle((top_left[0], top_left[1]), bw, bh, edgecolor='blue', facecolor='none', linewidth=2, zorder=10)
        ax.add_patch(rect)            
        # 图B上中心点 + 宽度偏移
        pt_b = centers1[idx_b]
        pt_b_shifted = (pt_b[0] + img0.shape[1], pt_b[1])

        # 画红色虚线连接
        ax.plot([cx, pt_b_shifted[0]], [cy, pt_b_shifted[1]], color='blue',  linewidth=1)
        
        # 图B上红点
        ax.scatter(*pt_b_shifted, color='blue', s=10, marker='o')



def add_border(img, border_thickness=5, color=(0, 0, 0)):
    """给图像加黑色边框"""
    return cv2.copyMakeBorder(
        img,
        border_thickness, border_thickness, border_thickness, border_thickness,
        cv2.BORDER_CONSTANT,
        value=color
    )
# def plot_yinshe_zuoyou(match00, match_filt, match_new, image0, image1, boxes0, boxes1, output_folder, img_file_a,buchong_id, buchong_bbox):
def plot_yinshe_zuoyou(match00, match_filt, match_new, image0, image1, boxes0, boxes1, output_folder, img_file_a,MDA,Precision,Recall,time,buchong_id, buchong_bbox):

    # 确保输出路径存在
    os.makedirs(output_folder, exist_ok=True)
    # 文件名去掉扩展名
    filename_base = os.path.splitext(os.path.basename(img_file_a))[0]  # e.g. '00001'
    ###############################
    # -------- 图 1：filt --------#
    ###############################
    img0 = tensor_to_npimg(image0).copy()
    img1 = tensor_to_npimg(image1).copy()

    draw_boxes(img0, boxes0)
    draw_boxes(img1, boxes1)
    # 加黑色边框这是1920×1080的
    border_thickness=8
    gap_size=10
    #track3数据集
    # border_thickness=2
    # gap_size=4
    img0 = add_border(img0, border_thickness)
    img1 = add_border(img1, border_thickness)
    # 中间插入空白
    gap = np.ones((img0.shape[0], gap_size, 3), dtype=np.uint8) * 255  # 白色空白，可改成 0 变成黑色
    img_combined1 = np.concatenate([img0, gap, img1], axis=1)
    fig1, ax1 = plt.subplots(figsize=(12,6))

    ax1.imshow(img_combined1, alpha=1,zorder=0)  # 整体透明度#############################################
    ax1.axis("off")

    true_match_idx=panduan(match00,boxes0,boxes1)

    match00_set = set(map(tuple, match00))
    match_filt_set = set(map(tuple, match_filt))
    removed = list(match00_set - match_filt_set)

    ##############
    ##修改boxes###
    #############
    # 修改boxes坐标：加上边框偏移
    boxes0_offset = []
    for box in boxes0:
        new_box = box.copy()
        new_box[2] += border_thickness
        new_box[3] += border_thickness   # cx + 左图边框
        boxes0_offset.append(new_box)

    boxes1_offset = []
    for box in boxes1:
        new_box = box.copy()
        new_box[2] += (2*border_thickness )
        new_box[3] += border_thickness  # cx + 右图边框
        boxes1_offset.append(new_box)


    # 找出 match_new 中不在 match_filt 的索引
    removed_idx= [i for i, m in enumerate(match_new) if tuple(m) not in match_filt_set]
    removed_idx = [i for i, m in enumerate(match00) if tuple(m) not in match_filt_set]
#   true_match_idx=panduan(match00,boxes0,boxes1)

    plot_matches(img0, img1, boxes0_offset, boxes1_offset, match00,removed_idx, true_match_idx,ax=ax1)

    save_path1 = os.path.join(output_folder, f"{filename_base}_1filt.png")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path1, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig1)

    #################################
    # -------- 图 2：buchong --------#
    #################################
    img0 = tensor_to_npimg(image0).copy()
    img1 = tensor_to_npimg(image1).copy()
    draw_boxes(img0, boxes0)
    draw_boxes(img1, boxes1)




    img0 = add_border(img0, border_thickness)
    img1 = add_border(img1, border_thickness)
    gap = np.ones((img0.shape[0], gap_size, 3), dtype=np.uint8) * 255
    img_combined2 = np.concatenate([img0, gap, img1], axis=1)

    fig2, ax2 = plt.subplots(figsize=(12,6))
    plot_unmatched_points_and_boxes(ax2, img0, boxes1_offset, buchong_id, buchong_bbox)  ############补充映射

    ax2.imshow(img_combined2,alpha=1,zorder=0)########################################################
    ax2.axis("off")
    # 在左上角添加文字（无背景）
    # text_str = f"MDA: {MDA:.3f}\nPrecision: {Precision:.3f}\nRecall: {Recall:.3f}\nTime: {time:.1f} ms"
    text_str = f"MDA: {MDA:.3f}\nPrecision: {Precision:.3f}\nRecall: {Recall:.3f}\n"
    # ax2.text(
    #     10, 20, text_str, 
    #     color='white', fontsize=12, weight='bold',
    #     ha='left', va='top'
    # )#源图是12字体


    match_new_set = set(map(tuple, match_new))
    new_add = list(match_new_set - match_filt_set)
    true_match_new=panduan(match_new,boxes0,boxes1)
    plot_matches_2(img0, img1,boxes0_offset, boxes1_offset, match_new,true_match_new,  ax=ax2)


    save_path2 = os.path.join(output_folder, f"{filename_base}_2buchong.png")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path2, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig2)

    print(f"✅ 匹配图保存至：\n  {save_path1}\n  {save_path2}")





##################################################################################################################################################################################################

def plot_matches_shangxia(image0, image1, boxes0, boxes1, matches, removed_idx, true_match_idx, ax):
    centers0 = get_centers(boxes0)
    centers1 = get_centers(boxes1)

    for k, (i, j) in enumerate(matches):
        pt1 = centers0[i]
        pt2 = (centers1[j][0] , centers1[j][1]+ image0.shape[0])

        # 线的颜色
        if k in true_match_idx:
            line_color = 'yellow'
        else:
            line_color = 'red'
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=line_color, linewidth=1, zorder=1)

        # 散点颜色
        if k in removed_idx:
            scatter_color = 'blue'
            marker = 'x'
            size = 40
            lw = 1.5
        elif k in true_match_idx:
            scatter_color = 'yellow'
            marker = 'o'
            size = 6
            lw = None
        else:
            scatter_color = 'red'
            marker = 'o'
            size = 6
            lw = None
        
        ax.scatter(*pt1, color=scatter_color, marker=marker, s=size, linewidths=lw, zorder=2)
        ax.scatter(*pt2, color=scatter_color, marker=marker, s=size, linewidths=lw, zorder=2)

def plot_matches_2_shangxia(image0, image1, boxes0, boxes1, matches, true_match_idx, ax):
    centers0 = get_centers(boxes0)
    centers1 = get_centers(boxes1)

    for k, (i, j) in enumerate(matches):
        pt1 = centers0[i]
        pt2 = (centers1[j][0], centers1[j][1] + image0.shape[0])

        if k in true_match_idx:  # ✅ 正确匹配
            color = 'yellow'
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=1)
            ax.scatter(*pt1, color=color, s=6, zorder=2)
            ax.scatter(*pt2, color=color, s=6, zorder=2)

        else:  # ❌ 错误匹配
            color = 'red'
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='red', linewidth=1)
            ax.scatter(*pt1, color=color, s=6, zorder=2)
            ax.scatter(*pt2, color=color, s=6, zorder=2)

        # 在中点画蓝色 ×
        # ax.scatter(mid_x, mid_y, color='blue', marker='x', s=20, linewidths=1.2,zorder=999)    
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_yinshe_shangxia(
    match00, match_filt, match_new,
    image0, image1, boxes0, boxes1,
    output_folder, img_file_a,
    MDA, Precision, Recall, time,
    buchong_id, buchong_bbox
):
    """绘制上下拼接和左右拼接的匹配结果"""

    os.makedirs(output_folder, exist_ok=True)
    filename_base = os.path.splitext(os.path.basename(img_file_a))[0]

    # === 公共参数 ===
    border_thickness = 8
    gap_size = 10

    # ========== 图1：上下拼接 (filt) ==========
    img0 = tensor_to_npimg(image0).copy()
    img1 = tensor_to_npimg(image1).copy()
    draw_boxes(img0, boxes0)
    draw_boxes(img1, boxes1)

    # 加边框
    img0 = add_border(img0, border_thickness)
    img1 = add_border(img1, border_thickness)

    # 插入 gap（水平拼接时 gap 在 axis=0，需要宽度相同）
    gap = np.ones((gap_size, img0.shape[1], 3), dtype=np.uint8) * 255
    img_combined1 = np.concatenate([img0, gap, img1], axis=0)

    fig1, ax1 = plt.subplots(figsize=(6, 12))
    ax1.imshow(img_combined1, alpha=1, zorder=0)
    ax1.axis("off")

    # 计算匹配
    true_match_idx = panduan(match00, boxes0, boxes1)
    match00_set = set(map(tuple, match00))
    match_filt_set = set(map(tuple, match_filt))
    removed_idx = [i for i, m in enumerate(match00) if tuple(m) not in match_filt_set]

    # 修正 boxes 偏移
    boxes0_offset = []
    for box in boxes0:
        new_box = box.copy()
        new_box[2] += border_thickness
        new_box[3] += border_thickness
        boxes0_offset.append(new_box)

    boxes1_offset = []
    for box in boxes1:
        new_box = box.copy()
        new_box[2] += border_thickness
        new_box[3] += border_thickness  # 保持一致，不要多加
        boxes1_offset.append(new_box)

    # 绘制
    plot_matches_shangxia(img0, img1, boxes0_offset, boxes1_offset,
                          match00, removed_idx, true_match_idx, ax=ax1)

    save_path1 = os.path.join(output_folder, f"{filename_base}_1filt.png")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path1, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig1)

    # ========== 图2：上下拼接 (buchong) ==========
    img0 = tensor_to_npimg(image0).copy()
    img1 = tensor_to_npimg(image1).copy()
    draw_boxes(img0, boxes0)
    draw_boxes(img1, boxes1)

    img0 = add_border(img0, border_thickness)
    img1 = add_border(img1, border_thickness)

    # 插入 gap（上下拼接，axis=0）
    gap = np.ones((gap_size, img0.shape[1], 3), dtype=np.uint8) * 255
    img_combined2 = np.concatenate([img0, gap, img1], axis=0)

    fig2, ax2 = plt.subplots(figsize=(6, 12))
    ax2.imshow(img_combined2, alpha=1, zorder=0)
    ax2.axis("off")


    # === 绘制匹配 ===
    true_match_new = panduan(match_new, boxes0, boxes1)
    plot_matches_2_shangxia(img0, img1, boxes0_offset, boxes1_offset,
                            match_new, true_match_new, ax=ax2)

    save_path2 = os.path.join(output_folder, f"{filename_base}_2buchong.png")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path2, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig2)



