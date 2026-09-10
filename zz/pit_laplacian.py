import torch
import torch.nn.functional as F
from torch.linalg import eigh  # PyTorch 1.9+ 支持对称矩阵特征分解
import torch.nn as nn


def compute_laplacian_pe_batch(kpts, knn=6, k_eig=6):
    """
    计算一批关键点的局部拉普拉斯谱编码
    
    参数：
        kpts: Tensor, [B, N, 2]，关键点坐标
        knn: int, 邻域大小
        k_eig: int, 提取特征值个数
        
    返回：
        lap_pe: Tensor, [B, N, k_eig]，每个点的拉普拉斯谱特征
    """
    B, N, _ = kpts.shape
    device = kpts.device
    lap_pe = torch.zeros(B, N, k_eig, device=device)


    for b in range(B):
        points = kpts[b]  # [N, 2]

        # 计算点间距离矩阵，形状[N, N]
        dist_mat = torch.cdist(points, points, p=2)  # Euclidean distance

        # 对每个点找到knn邻居索引，包含自己
        knn_dists, knn_idxs = torch.topk(-dist_mat, k=knn + 1, largest=True)  # negative dist求最近
        knn_idxs = knn_idxs[:, 1:]  # 去除自己，形状 [N, knn]

        for i in range(N):
            neighbors_idx = knn_idxs[i]  # [knn]
            neighbors = points[neighbors_idx]  # [knn, 2]
            center = points[i].unsqueeze(0)  # [1, 2]

            # 邻居包含中心点自己
            sub_points = torch.cat([center, neighbors], dim=0)  # [knn+1, 2]

            # 计算局部距离矩阵
            sub_dist = torch.cdist(sub_points, sub_points, p=2)  # [knn+1, knn+1]

            # 构造邻接矩阵，使用高斯核权重
            sigma = torch.mean(sub_dist[0, 1:]) + 1e-8  # 中心点到邻居的平均距离
            A = torch.exp(- (sub_dist ** 2) / (2 * sigma ** 2))
            A.fill_diagonal_(0)

            # 计算度矩阵D
            D = torch.diag(A.sum(dim=1))

            # 计算归一化拉普拉斯矩阵L = I - D^{-1/2} A D^{-1/2}
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(torch.diag(D), min=1e-8)))
            L = torch.eye(knn + 1, device=device) - D_inv_sqrt @ A @ D_inv_sqrt

            # 计算L的前k_eig个最小特征值
            eigvals, _ = eigh(L)  # eigvals升序排列
            lap_pe[b, i] = eigvals[:k_eig]


    return lap_pe  # [B, N, k_eig]

def compute_similarity_matrix(feat0, feat1):
    """
    计算两组拓扑特征之间的相似度矩阵，形状[B, N, N]

    这里用负欧式距离的指数作为相似度，也可以用余弦相似度

    参数：
        feat0, feat1: Tensor, [B, N, k_eig]
    返回：
        sim_mat: Tensor, [B, N, N]
    """
    # 归一化特征，便于相似度计算
    feat0_norm = F.normalize(feat0, dim=-1)
    feat1_norm = F.normalize(feat1, dim=-1)

    sim_mat = torch.bmm(feat0_norm, feat1_norm.transpose(1, 2))  # [B, N, N]
    return sim_mat

# ==== 用法示例 ====
batch_size = 2
num_points = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

kpts0 = torch.rand(batch_size, num_points, 2, device=device)
kpts1 = torch.rand(batch_size, num_points, 2, device=device)

lap_pe0 = compute_laplacian_pe_batch(kpts0, knn=10, k_eig=6)
lap_pe1 = compute_laplacian_pe_batch(kpts1, knn=10, k_eig=6)

similarity_matrix = compute_similarity_matrix(lap_pe0, lap_pe1)  # [B, N, N]

print(similarity_matrix.shape)  # torch.Size([2, 100, 100])
