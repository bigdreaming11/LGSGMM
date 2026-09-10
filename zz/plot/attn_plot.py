import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.colors as mcolors
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import numpy as np
import os
import matplotlib.pyplot as plt

def add_border(img, border_thickness=5, color=(0, 0, 0)):
    """给图像加黑色边框"""
    import cv2
    return cv2.copyMakeBorder(
        img,
        border_thickness, border_thickness, border_thickness, border_thickness,
        cv2.BORDER_CONSTANT,
        value=color
    )



def draw_box(ax, box, color='lime'):
    x_c, y_c, w, h = box[2:6]
    x = x_c - w / 2
    y = y_c - h / 2
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

# def draw_connection(ax, pt1, pt2, weight, color='red', threshold=0.000005):
#     if weight < threshold:
#         return
    
#     # 平滑增长透明度（非线性），越大越接近1.0
#     alpha = weight ** (1/4)
#     alpha = min(max(alpha, 0.1), 1.0)

#     # 线宽也跟权重相关
#     linewidth = 1+ weight * 8 # 控制粗细范围
    
#     ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, alpha=alpha, linewidth=linewidth)




def draw_connection(ax, pt1, pt2, weight_norm, threshold=0.000005):
    if weight_norm < threshold:
        return

    # 线宽保持原公式
    linewidth = 1 + weight_norm * 2

    # 定义颜色 (RGB 0~1)
    deep_red = np.array([139/255, 0, 0])
    red = np.array([1.0, 0, 0])
    light_red = np.array([1.0, 181/255, 197/255])

    # 颜色映射
    if weight_norm < 0.1:
        t = weight_norm / 0.1
        color = (1 - t) * red + t * light_red   # 红 -> 浅红渐变
        # alpha 变化慢一些，可用开方/三次方根减缓
        alpha = 0.2 + 0.8 * (weight_norm / 0.1) ** (1)
    else:
        color = deep_red if weight_norm > 0.7 else red
        alpha = 1.0  # 中大权重不透明

    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=linewidth, alpha=alpha)






def compute_box_centers(boxes):
    return np.array([[b[2], b[3]] for b in boxes])

def plot_self_attn(self_attn0, image0, boxes0, idx, img_file_a, save_dir, dpi=100, threshold=0.05):
    os.makedirs(save_dir, exist_ok=True)

    centers = compute_box_centers(boxes0)
    query_center = centers[idx]
    attn_layers = len(self_attn0)

    for i in range(attn_layers):
        fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

        # 去掉背景色
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        # 正确处理图像像素范围
        img_np = image0.permute(1, 2, 0).cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = img_np.clip(0, 255).astype(np.uint8)

        ax.imshow(img_np)

        # # 左上角文字（保持原样式，只是位置改左上角）
        # ax.text(
        #     0.01, 0.99, f'Self Attention Layer {i + 1}',
        #     color='white', fontsize=16, fontweight='bold',
        #     ha='left', va='top', transform=ax.transAxes
        # )


        # 绘制绿色框
        for box in boxes0:
            draw_box(ax, box, color='lime')

        # 平均多头注意力
        attn = self_attn0[i][0].mean(0)  # [n, n]

        # 收集有效权重，计算 min/max
        weights_all = [attn[idx, j].item() for j in range(len(centers)) if j != idx and attn[idx, j].item() >= threshold]
        w_min, w_max = min(weights_all), max(weights_all)

        for j, center in enumerate(centers):
            if j == idx:
                continue
            weight = attn[idx, j].item()
            if weight >= threshold:
                # 按相对大小归一化
                weight_norm = (weight - w_min) / (w_max - w_min + 1e-8)
                draw_connection(ax, query_center, center, weight_norm, threshold)
        # 去掉刻度但保留边框
        # 去掉刻度但保留边框
        ax.set_xticks([])
        ax.set_yticks([])

        # 黑色边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(3)

        # 调整子图边界，保证没有额外填充
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        save_path = os.path.join(save_dir, f'{os.path.splitext(img_file_a)[0]}_lay{i+1}.png')
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        print(f'[✓] Saved self-attn layer {i+1} to {save_path}')





def plot_cross_attn(cross_attn01, image0, image1, boxes0, boxes1, idx, img_file_a,
                    save_dir='attn_cross_vis', dpi=200, threshold=0.05, gap_size=10, border_thickness=1):
    import matplotlib.patches as patches
    os.makedirs(save_dir, exist_ok=True)

    centers0 = compute_box_centers(boxes0)
    centers1 = compute_box_centers(boxes1)
    query_center = centers0[idx]
    attn_layers = len(cross_attn01)

    for i in range(attn_layers):
        # 转 numpy
        img0_np = image0.permute(1,2,0).cpu().numpy()
        img1_np = image1.permute(1,2,0).cpu().numpy()
        img0_np = (img0_np*255).clip(0,255).astype(np.uint8) if img0_np.max()<=1 else img0_np.clip(0,255).astype(np.uint8)
        img1_np = (img1_np*255).clip(0,255).astype(np.uint8) if img1_np.max()<=1 else img1_np.clip(0,255).astype(np.uint8)

        # 中间白色 gap
        gap = np.ones((img0_np.shape[0], gap_size, img0_np.shape[2]), dtype=np.uint8) * 255
        canvas = np.concatenate([img0_np, gap, img1_np], axis=1)

        # 根据 canvas 尺寸设置 fig
        fig_height = canvas.shape[0]/dpi
        fig_width = canvas.shape[1]/dpi
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        # 去掉背景
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        ax.imshow(canvas)

        # 左右黑框
        ax.add_patch(patches.Rectangle((0,0), img0_np.shape[1], img0_np.shape[0],
                                       linewidth=border_thickness, edgecolor='black', facecolor='none'))
        ax.add_patch(patches.Rectangle((img0_np.shape[1]+gap_size,0), img1_np.shape[1], img1_np.shape[0],
                                       linewidth=border_thickness, edgecolor='black', facecolor='none'))

        # 绘制 boxes
        for box in boxes0:
            draw_box(ax, box, color='lime')
        offset_x = img0_np.shape[1] + gap_size
        for box in boxes1:
            b = box.copy()
            b[2] += offset_x
            draw_box(ax, b, color='lime')

        # cross-attn 线
        attn = cross_attn01[i][0].mean(0)
        weights_all = [attn[idx,j].item() for j in range(len(centers1)) if attn[idx,j].item() >= threshold]
        w_min, w_max = (min(weights_all), max(weights_all)) if weights_all else (0,1)
        for j, center1 in enumerate(centers1):
            weight = attn[idx,j].item()
            if weight >= threshold:
                weight_norm = (weight - w_min)/(w_max - w_min + 1e-8)
                pt1 = query_center
                pt2 = center1 + np.array([offset_x, 0])
                draw_connection(ax, pt1, pt2, weight_norm, threshold)

        ax.axis('off')
        # 关键：强制设置 xlim 和 ylim 与 canvas 像素严格对齐
        ax.set_xlim(0, canvas.shape[1])
        ax.set_ylim(canvas.shape[0], 0)  # 注意：imshow 默认 origin='upper'，y 轴向下

        # 完全移除所有边距
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        save_path = os.path.join(save_dir, f'{os.path.splitext(img_file_a)[0]}+lay{i+1}.png')
        plt.savefig(
            save_path,
            dpi=dpi,
            transparent=True,
            pad_inches=0,          # 显式禁止 padding
            bbox_inches='tight'    # 注意：这里看似矛盾，但配合 pad_inches=0 可消除边缘空白
        )
        plt.close()
        print(f'[✓] Saved cross-attn layer {i+1} to {save_path}')



# def plot_cross_attn(cross_attn01, image0, image1, boxes0, boxes1, idx, img_file_a,save_dir='attn_cross_vis', dpi=100, threshold=0.05):
#     os.makedirs(save_dir, exist_ok=True)

#     centers0 = compute_box_centers(boxes0)
#     centers1 = compute_box_centers(boxes1)
#     query_center = centers0[idx]
#     w0 = image0.shape[2]
#     attn_layers = len(cross_attn01)

#     for i in range(attn_layers):
#         fig, ax = plt.subplots(figsize=(16, 6), dpi=dpi)

#         # 图像格式转换
#         img0_np = image0.permute(1, 2, 0).cpu().numpy()
#         img1_np = image1.permute(1, 2, 0).cpu().numpy()

#         if img0_np.max() <= 1.0:
#             img0_np = (img0_np * 255).clip(0, 255).astype(np.uint8)
#         else:
#             img0_np = img0_np.clip(0, 255).astype(np.uint8)

#         if img1_np.max() <= 1.0:
#             img1_np = (img1_np * 255).clip(0, 255).astype(np.uint8)
#         else:
#             img1_np = img1_np.clip(0, 255).astype(np.uint8)

#         canvas = np.concatenate([img0_np, img1_np], axis=1)
#         ax.imshow(canvas)
#         # ax.set_title(f'Cross Attention Layer {i + 1}')
#         # ax.text(
#         #         0.01, 0.99, f'Cross Attention Layer {i + 1}',
#         #         color='white', fontsize=16, fontweight='bold',
#         #         ha='left', va='top', transform=ax.transAxes
#         #     )

#         # 画绿色框（图0）
#         for box in boxes0:
#             draw_box(ax, box, color='lime')

#         # 画绿色框（图1，右图，要偏移）
#         for box in boxes1:
#             x_c, y_c, w, h = box[2:6]
#             x_shifted = x_c - w / 2 + w0
#             y_shifted = y_c - h / 2
#             rect = patches.Rectangle((x_shifted, y_shifted), w, h, linewidth=1, edgecolor='lime', facecolor='none')
#             ax.add_patch(rect)

#         # 获取该层 cross attention 矩阵并平均多头
#         attn = cross_attn01[i][0].mean(0)  # shape: [num_queries, num_keys]

#         # 收集有效权重
#         weights_all = [attn[idx, j].item() for j in range(len(centers1)) if attn[idx, j].item() >= threshold]
#         if not weights_all:
#             continue
#         w_min, w_max = min(weights_all), max(weights_all)

#         # 画注意力连线
#         for j, center1 in enumerate(centers1):
#             weight = attn[idx, j].item()
#             if weight >= threshold:
#                 # 相对归一化
#                 weight_norm = (weight - w_min) / (w_max - w_min + 1e-8)
#                 pt2 = [center1[0] + w0, center1[1]]
#                 draw_connection(ax, query_center, pt2, weight_norm, threshold)

#         ax.axis('off')

#         save_path = os.path.join(save_dir, f'{os.path.splitext(img_file_a)[0]}+lay{i+1}.png')
#         plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
#         plt.close()
#         print(f'[✓] Saved cross-attn layer {i+1} to {save_path}')
