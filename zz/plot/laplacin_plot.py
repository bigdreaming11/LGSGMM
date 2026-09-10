import torch
import matplotlib.pyplot as plt
import os
def laplacin_plot(lap_pe0, batch_idx=0, point_idx=10,save_dir_laplacin=None,filename=None):
    spectra = lap_pe0[batch_idx]  # shape: (num, 6)
    num_points = spectra.shape[0]  
    neighbor_indices = [point_idx - 2,point_idx,  point_idx + 2]
    # neighbor_indices = [point_idx - 4, point_idx - 7, point_idx, point_idx + 2, point_idx + 2]
    x = range(1,spectra.shape[1])  # 横坐标: 0,1,2,3,4,5
    plt.figure(figsize=(6,4))
    # for idx in neighbor_indices:
    #     y = spectra[idx].cpu().numpy()[1:]
    #     plt.plot(x, y, marker='o', label=f"point {idx}")    
    step =20  # 每隔几个点画一次
    x = range(1, spectra.shape[1], step)
    for idx in neighbor_indices:
        y = spectra[idx].cpu().numpy()[1::step]  # 同样步长采样
        plt.plot(x, y, marker='o', label=f"point {idx}")
    plt.xlabel("Spectral index")
    plt.ylabel("Spectral value")
    plt.title(f"Local spectral features around point {point_idx}")
    plt.legend()
    plt.grid(True)

    # 保存，不显示
    save_path = os.path.join(save_dir_laplacin, f"{filename}_{point_idx}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()




# import torch
# import matplotlib.pyplot as plt
# import os

# def laplacin_compare_plot(lap_pe0, lap_pe1, batch_idx=0, point_idx=8, save_dir_laplacin=None, filename=None, step=1):
#     """
#     同时绘制 lap_pe0 和 lap_pe1 的谱特征
#     lap_pe0/lap_pe1: [B, num, k] Tensor
#     point_idx: 当前关注点
#     step: 横坐标步长
#     """
#     spectra0 = lap_pe0[batch_idx]  # [num, k]
#     spectra1 = lap_pe1[batch_idx]  # [num, k]
#     num_points = spectra0.shape[0]

#     # 选择邻域点
#     neighbor_indices = [point_idx - 8,point_idx - 1, point_idx]
#     # if point_idx>num_points -1: 
#     #     neighbor_indices = [point_idx - 7, num_points -1]
#     #     if point_idx - 7>num_points -1:
#     #         neighbor_indices = [ num_points -2, num_points -1]

#     # 横坐标采样
#     x = range(1, spectra0.shape[1], step)

#     plt.figure(figsize=(12,6))

#     # 为每个点选择一个颜色
#     colors = plt.cm.tab10.colors  # 10种颜色循环
#     for i, idx in enumerate(neighbor_indices):
#         color = colors[i % len(colors)]
#         y0 = spectra0[idx].cpu().numpy()[1::step]
#         y1 = spectra1[idx].cpu().numpy()[1::step]
#         # lap_pe0 用圆形
#         plt.plot(x, y0, marker='o', linestyle='-', color=color, label=f"point {idx} (lap_pe0)")
#         # lap_pe1 用三角
#         plt.plot(x, y1, marker='^', linestyle='-', color=color, label=f"point {idx} (lap_pe1)")

#     plt.xlabel("Spectral index")
#     plt.ylabel("Spectral value")
#     plt.title(f"Compare spectral features around point {point_idx}")
#     plt.legend()
#     plt.grid(True)

#     # 保存
#     save_path = os.path.join(save_dir_laplacin, f"{filename}_{point_idx}.png")
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, dpi=100, bbox_inches="tight")
#     plt.close()

import numpy as np
import matplotlib.pyplot as plt
import os

def laplacin_compare_plot(lap_pe0, lap_pe1, batch_idx=0, point_idx=8, save_dir_laplacin=None, filename=None, step=1):
    """
    同时绘制 lap_pe0 和 lap_pe1 的谱特征（归一化到0-1）
    lap_pe0/lap_pe1: [B, num, k] Tensor
    point_idx: 当前关注点
    step: 横坐标步长
    """
    spectra0 = lap_pe0[batch_idx]  # [num, k]
    spectra1 = lap_pe1[batch_idx]  # [num, k]
    num_points = spectra0.shape[0]

    # 选择邻域点
    neighbor_indices = [point_idx - 5, point_idx]

    # 横坐标采样
    x = range(1, spectra0.shape[1], step)

    plt.figure(figsize=(12,6))

    # 为每个点选择一个颜色
    colors = plt.cm.tab10.colors  # 10种颜色循环
    for i, idx in enumerate(neighbor_indices):
        color = colors[i % len(colors)]
        y0 = spectra0[idx].cpu().numpy()[1::step]
        y1 = spectra1[idx].cpu().numpy()[1::step]

        # 归一化
        y0 = (y0 - y0.min()) / (y0.max() - y0.min() + 1e-8)  # 避免除以0
        y1 = (y1 - y1.min()) / (y1.max() - y1.min() + 1e-8)

        # # lap_pe0 用圆形
        if idx== point_idx:
            plt.plot(x, y0, marker='o', linestyle='-', color=color,     label=f"target0 img0")
            # lap_pe1 用三角
            plt.plot(x, y1, marker='^', linestyle='-', color=color, label=f"target0 img1")
        if idx== point_idx - 5:
            plt.plot(x, y0, marker='o', linestyle='-', color=color, label=f"target1 img0")
            # lap_pe1 用三角
            plt.plot(x, y1, marker='^', linestyle='-', color=color, label=f"target1 img1")            
        # lap_pe0 用圆形
        # if idx== point_idx:
        #     plt.plot(x, y0, marker='o', linestyle='-', color="orange",     label=f"target0 img0")
        #     # lap_pe1 用三角
        #     plt.plot(x, y1, marker='o', linestyle='-', color=, label=f"target0 img1")
        # if idx== point_idx - 5:
        #     plt.plot(x, y0, marker='^', linestyle='-', color=color, label=f"target1 img0")
        #     # lap_pe1 用三角
        #     plt.plot(x, y1, marker='^', linestyle='-', color=color, label=f"target1 img1")     
    plt.xlabel("Feature dimension",fontsize=14)
    plt.ylabel("Normalized spectral value",fontsize=14)
    # plt.title(f"Compare normalized spectral features around point {point_idx}")
    plt.legend(fontsize=14,loc='upper right',)
    plt.grid(True)

    # 保存
    save_path = os.path.join(save_dir_laplacin, f"{point_idx}_{filename}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# import numpy as np
# import matplotlib.pyplot as plt
# import os

# def laplacin_compare_plot(lap_pe0, lap_pe1, batch_idx=0, point_idx=8, save_dir_laplacin=None, filename=None, step=1):
#     """
#     同时绘制 lap_pe0 和 lap_pe1 的谱特征
#     lap_pe0/lap_pe1: [B, num, k] Tensor
#     point_idx: 当前关注点
#     step: 横坐标步长
#     """
#     spectra0 = lap_pe0[batch_idx]  # [num, k]
#     spectra1 = lap_pe1[batch_idx]  # [num, k]
#     num_points = spectra0.shape[0]

#     # 选择邻域点
#     neighbor_indices = [point_idx - 6, point_idx]

#     # 横坐标采样
#     x = range(1, spectra0.shape[1], step)

#     plt.figure(figsize=(12,6))

#     # 为每个点选择一个颜色
#     colors = plt.cm.tab10.colors  # 10种颜色循环
#     for i, idx in enumerate(neighbor_indices):
#         color = colors[i % len(colors)]
#         y0 = spectra0[idx].cpu().numpy()[1::step]
#         y1 = spectra1[idx].cpu().numpy()[1::step]

#         # # 根据不同 idx 添加不同噪声
#         # if idx == point_idx :
#         #     noise0 = np.random.uniform(-0.05, 0.1, size=y0.shape)
#         #     noise1 = np.random.uniform(-0.05,0.1, size=y1.shape)
#         #     y0 = y0 + noise0
#         #     y1 = y1 + noise1
#         # elif idx == point_idx - 2:
#         #     noise0 = np.random.uniform(-0.05, 0, size=y0.shape)
#         #     noise1 = np.random.uniform(-0.05, 0, size=y1.shape)
#         #     y0 = y0 + noise0
#         #     y1 = y1 + noise1
#         # idx == point_idx 不加噪声

#         # lap_pe0 用圆形
#         plt.plot(x, y0, marker='o', linestyle='-', color=color, label=f"point {idx} (lap_pe0)")
#         # lap_pe1 用三角
#         plt.plot(x, y1, marker='^', linestyle='-', color=color, label=f"point {idx} (lap_pe1)")

#     plt.xlabel("Spectral index")
#     plt.ylabel("Spectral value")
#     plt.title(f"Compare spectral features around point {point_idx}")
#     plt.legend()
#     plt.grid(True)

#     # 保存
#     save_path = os.path.join(save_dir_laplacin, f"{point_idx}_{filename}.png")
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, dpi=100, bbox_inches="tight")
#     plt.close()


