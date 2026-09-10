import numpy as np
import cv2  # 添加在文件顶部
# from zz.guss_guide_ransac import weighted_ransac
def center_of_box(box):
    x, y, w, h = box[2], box[3], box[4], box[5]
    return np.array([x , y ])

def cosine_similarity_matrix(features0, features1):
    # 计算M×N的余弦相似度矩阵
    norm0 = features0 / np.linalg.norm(features0, axis=1, keepdims=True)
    norm1 = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
    return norm0 @ norm1.T  # M×N

def transform_point(T, x):
    s, R, t = T
    return s * R.dot(x) + t

def initialize_transformation():
    s = 1.0
    R = np.eye(2)
    t = np.zeros(2)
    return (s, R, t)

def update_transformation_rigid(X, Y, P):
    """
    刚性变换参数更新（旋转矩阵R，缩放s，平移t）
    X: N×2，模型点
    Y: M×2，目标点
    P: M×N，后验概率矩阵
    """
    M, N = P.shape
    weights = P.sum(axis=0)  # 对N个模型点的权重求和，大小N
    total_weight = weights.sum()
    
    # 计算加权均值
    mu_x = (weights[:, None] * X).sum(axis=0) / total_weight  # N×2 -> 1×2
    mu_y = (P.T.dot(Y)).sum(axis=0) / total_weight            # M×2 -> N×2加权和 -> 1×2
    
    # 中心化点集
    Xc = X - mu_x
    Yw = P.T.dot(Y)   # N×2
    Yc = Yw - weights[:, None] * mu_y

    # 计算加权协方差矩阵
    W = Xc.T.dot(Yc)  # 2×2矩阵

    # 奇异值分解
    U, _, Vt = np.linalg.svd(W)
    R = Vt.T.dot(U.T)
    # 确保旋转矩阵行列式为1（正交矩阵且无反射）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T.dot(U.T)

    var_x = np.sum(weights * np.sum(Xc**2, axis=1))  # 加权X的方差
    
    s = np.trace(R.T.dot(W)) / var_x  # 缩放因子
    
    t = mu_y - s * R.dot(mu_x)        # 平移向量

    return (s, R, t)

def update_variance(X, Y, P, T):
    s, R, t = T
    M, N = P.shape
    numerator = 0.0
    denominator = 0.0
    for m in range(M):
        for n in range(N):
            diff = Y[m] - transform_point(T, X[n])
            numerator += P[m, n] * np.dot(diff, diff)
            denominator += 2 * P[m, n]
    return numerator / denominator if denominator > 0 else 1e-6

def update_outlier_ratio(P):
    return 1 - P.sum() / P.size

def fg_gmm_rigid_match(boxes0, boxes1, match, features0, features1, scores, max_iter=50, tol=1e-5, alpha=1, tau=0.9, threshold=0.5):
    """
    只考虑刚性变换的FG-GMM匹配算法（含半监督EM）
    """
    X = np.array([center_of_box(b) for b in boxes0])  # N×2
    Y = np.array([center_of_box(b) for b in boxes1])  # M×2
    M, N = len(Y), len(X)

    # 计算特征余弦相似度矩阵 (M×N)
    sim_mat = cosine_similarity_matrix(features1, features0)  # features1对应M，features0对应N

    # 构造初始匹配映射表
    known_matches = {m: n for (n, m) in match}

    # 初始化 π（M×N）
    pi = np.ones((M, N)) * ((1 - tau) / (N - 1))
    for (n, m), score in zip(match, scores):
        # pi[m, n] = tau  # 原始版本
        pi[m, n] = tau * score * sim_mat[m, n] #进一步引入相似度
        # pi[m, n] =  score #进一步引入相似度
    pi = pi / (pi.sum(axis=1, keepdims=True) + 1e-12)

    # 初始化参数
    T = initialize_transformation()
    # sigma2 = np.var(Y - X.mean(axis=0))

    # 论文建议：根据初始匹配对的距离初始化 sigma2，更稳健
    if len(match) > 0:
        matched_x = np.array([center_of_box(boxes0[n]) for (n, m) in match])
        matched_y = np.array([center_of_box(boxes1[m]) for (n, m) in match])
        diffs = matched_y - matched_x
        sigma2 = np.sum(np.linalg.norm(diffs, axis=1) ** 2) / (2 * matched_x.shape[0] * 2)
    else:
        sigma2 = 1.0  # fallback

    gamma = 0.1
    uniform_area = np.prod(Y.max(axis=0) - Y.min(axis=0))

    for iter_idx in range(max_iter):
        T_old = T
        P = np.zeros((M, N))  # 后验概率矩阵

        for m in range(M):
            if m in known_matches:
                # Eq. (5): 半监督 - 用 pi 直接指定
                n = known_matches[m]
                P[m, :] = 1e-10  # 其他位置给极小值防止全0
                P[m, n] = pi[m, n]
            else:
                # Eq. (6): soft assignment
                numerators = np.zeros(N)
                for n in range(N):
                    diff = Y[m] - transform_point(T, X[n])
                    dist_exp = np.exp(-np.linalg.norm(diff)**2 / (2 * sigma2))
                    numerators[n] = dist_exp * (pi[m, n] ** alpha)
                denom = numerators.sum() + gamma / uniform_area
                P[m, :] = numerators / denom if denom > 0 else numerators

        # M 步
        T = update_transformation_rigid(X, Y, P)
        sigma2 = update_variance(X, Y, P, T)
        gamma = update_outlier_ratio(P)

        # 收敛判断
        rot_diff = np.linalg.norm(T[1] - T_old[1])
        if rot_diff < tol:
            print(f"迭代终止: 第{iter_idx+1}次达到收敛阈值")
            break

    # 最后一轮对所有点重新 soft assignment（Eq.6）
    P_final = np.zeros((M, N))
    for m in range(M):
        numerators = np.zeros(N)
        for n in range(N):
            diff = Y[m] - transform_point(T, X[n])
            dist_exp = np.exp(-np.linalg.norm(diff)**2 / (2 * sigma2))
            numerators[n] = dist_exp * (pi[m, n] ** alpha)
        denom = numerators.sum() + gamma / uniform_area
        P_final[m, :] = numerators / denom if denom > 0 else numerators

    # 根据后验概率筛选匹配
    match_gmm = []
    for m in range(M):
        n_best = np.argmax(P_final[m])
        if P_final[m, n_best] > threshold:
            match_gmm.append([n_best, m])



        # 提取用于单应矩阵估计的点对
    src_pts = np.array([X[n] for n, m in match_gmm])
    dst_pts = np.array([Y[m] for n, m in match_gmm])


    # filtered_matches, H_final=weighted_ransac(boxes0, boxes1, P_final, threshold=50, max_trials=2000)

    # 估计单应矩阵（使用 RANSAC）
    # if len(src_pts) >= 4:
    #     H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=50)
    # else:
    #     H = None
    #     print("匹配点不足，无法估计单应矩阵。")



    return match_gmm,P_final 

