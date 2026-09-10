from modulefinder import Module
import torch
import torch.nn as nn


# class MatchingLoss(nn.Module):
#     def __init__(self, margin=1, alpha=0.6, beta=0.3, gamma=0.1):#margin默认1
#         """
#         MatchingLoss 计算三部分损失：
#         1. 交叉熵损失 (L1) - 计算匹配目标框的类别和 ID 误差。
#         2. 三元组损失 (L2) - 计算匹配目标框的特征向量间的距离。
#         3. 覆盖损失 (L3) - 让更多目标框被匹配，匹配分数更高。

#         参数：
#         - margin: Triplet loss 的 margin 超参数。
#         - alpha: L1 交叉熵损失的权重。
#         - beta: L2 三元组损失的权重。
#         - gamma: L3 覆盖损失的权重。
#         """
#         super(MatchingLoss, self).__init__()
#         self.margin = margin
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
#         self.BCELOSS = torch.nn.BCELoss()

#     def forward(self, matches, scores, features_a, features_b, boxes_a, boxes_b):
#         """
#         计算匹配损失。

#         参数：
#         - matches: List[Tensor], 每个样本的匹配索引，形状为 (N_i, 2)。
#         - scores: List[Tensor], 每个样本的匹配置信度分数，形状为 (N_i,)。
#         - features_a: List[Tensor], 每个样本的第一个图像的关键点特征，形状为 (M_i, D)。
#         - features_b: List[Tensor], 每个样本的第二个图像的关键点特征，形状为 (K_i, D)。
#         - boxes_a: List[Tensor], 每个样本的第一个图像的目标框信息，形状为 (M_i, 2)。
#         - boxes_b: List[Tensor], 每个样本的第二个图像的目标框信息，形状为 (K_i, 2)。

#         返回：
#         - loss_total: 总损失。
#         """
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         batch_losses = []

#         for b in range(len(matches)):  # 遍历每个样本
#             match_indices = matches[b]  # (N_i, 2)
#             score = scores[b]  # (N_i,)
#             feature_a = features_a[b].to(device)  # (M_i, D)
#             feature_b = features_b[b].to(device)  # (K_i, D)  # (K_i, 2)
#             box_a = boxes_a[b].to(device)  # 转换为张量并移动到设备
#             box_b = boxes_b[b].to(device)

#             # ========== 修正特征数量 ==========
#             # feature_a 多于 box_a → 裁掉尾部 padding
#             if feature_a.shape[0] > box_a.shape[0]:
#                 feature_a = feature_a[: box_a.shape[0]]

#             # feature_b 同理
#             if feature_b.shape[0] > box_b.shape[0]:
#                 feature_b = feature_b[: box_b.shape[0]]
#             if len(match_indices) == 0:  # 如果没有匹配，跳过该样本
#                 continue
            
#             # 提取匹配的类别和 ID
#             idx_a, idx_b = match_indices[:, 0], match_indices[:, 1]

#             #过滤
#             idx_a, idx_b = match_indices[:, 0], match_indices[:, 1]
#             valid_mask = (idx_a >= 0) & (idx_a < box_a.shape[0]) & (idx_b >= 0) & (idx_b < box_b.shape[0])
#             idx_a = idx_a[valid_mask].to(box_a.device)
#             idx_b = idx_b[valid_mask].to(box_b.device)
#             score = score[valid_mask].to(device) 

# #########################################################
#             class_a = box_a[idx_a, 0]  # (N_i,)
#             id_a = box_a[idx_a, 1]  # (N_i,)
#             class_b = box_b[idx_b, 0]
#             id_b = box_b[idx_b, 1]  # (N_i,)

#             # 计算匹配标签
#             labels = ((class_a == class_b) & (id_a == id_b)).float().to(device)  # (N_i,)

#             # === L1: 交叉熵损失 ===
#             score = score.to(device)
#             loss_id = self.BCELOSS(score, labels)

#             # === L2: 三元组损失 ===
#             # === L2: 三元组损失 ===
#             feat_a = feature_a[idx_a]  # (N_i, D)
#             feat_b = feature_b[idx_b]  # (N_i, D)

#             # === L2: Triplet Loss based on all features ===

#             anchors = []
#             positives = []
#             negatives = []

#             for i in range(len(box_a)):   # 遍历所有 A 中的框
#                 anchor = feature_a[i]     # (2048,)
#                 aid = box_a[i, 1].item()  # A 中当前 anchor 的 ID

#                 # ----- Positive candidates：B 中 ID == aid -----
#                 pos_mask = (box_b[:, 1] == aid)
#                 pos_candidates = feature_b[pos_mask]  # shape (P, 2048)

#                 if len(pos_candidates) == 0:
#                     continue

#                 # ----- Negative candidates：B 中 ID != aid -----
#                 neg_mask = (box_b[:, 1] != aid)
#                 neg_candidates = feature_b[neg_mask]  # shape (N, 2048)

#                 if len(neg_candidates) == 0:
#                     continue

#                 # ---- 随机选一个正样本 ----
#                 rand_p = torch.randint(0, len(pos_candidates), (1,))
#                 positive = pos_candidates[rand_p]

#                 # ---- 随机选一个负样本 ----
#                 rand_n = torch.randint(0, len(neg_candidates), (1,))
#                 negative = neg_candidates[rand_n]

#                 # ---- 存入 list ----
#                 anchors.append(anchor.reshape(-1))
#                 positives.append(positive.reshape(-1))
#                 negatives.append(negative.reshape(-1))

#             # === Triplet Loss ===
#             if len(anchors) > 0:
#                 A = torch.stack(anchors)  # (T, 2048)
#                 P = torch.stack(positives)
#                 N = torch.stack(negatives)
#                 loss_triplet = self.triplet_loss(A, P, N)
#             else:
#                 loss_triplet = torch.tensor(0.0, device=device)


#             # === L3: 覆盖损失 ===
#             # 惩罚未匹配的目标框
#             matched_a = torch.zeros(len(box_a), device=device)
#             matched_b = torch.zeros(len(box_b), device=device)
#             # matched_a[idx_a] = 1  # 标记已匹配
#             # matched_b[idx_b] = 1  # 标记已匹配
#             # loss_unmatched = (1 - matched_a).sum() + (1 - matched_b).sum()

#             loss_unmatched = len(matched_a)-len(idx_a)+len(matched_b)-len(idx_b)

#             # 惩罚低分匹配 (鼓励高分匹配)
#             loss_confidence = -torch.log(score + 1e-6).mean()  # 避免 log(0)

#             loss_cover = 0.02 * loss_unmatched + loss_confidence

#             # === 总损失 ===
#             # self.beta=0
#             # print(f"身份损失：{loss_triplet:>8.3f}, 三元组损失：{loss_triplet:>8.3f}, 惩罚损失：{self.gamma * loss_cover:>8.3f}")

#             loss_total = self.alpha * loss_id + self.beta * loss_triplet + self.gamma * loss_cover
#             batch_losses.append(loss_total)

#         # 如果所有样本都没有有效匹配，返回 0 损失
#         if len(batch_losses) == 0:
#             return torch.tensor(0.0, device=device, requires_grad=True)

#         # 对所有样本的损失求平均
#         total_loss = torch.mean(torch.stack(batch_losses))
#         return total_loss

# ###############################################修改版本2#########################
# # class MatchingLoss(nn.Module):
# #     def __init__(self, margin=1, alpha=0.6, beta=0.3, gamma=0.1):#margin默认1
# #         """
# #         MatchingLoss 计算三部分损失：
# #         1. 交叉熵损失 (L1) - 计算匹配目标框的类别和 ID 误差。
# #         2. 三元组损失 (L2) - 计算匹配目标框的特征向量间的距离。
# #         3. 覆盖损失 (L3) - 让更多目标框被匹配，匹配分数更高。

# #         参数：
# #         - margin: Triplet loss 的 margin 超参数。
# #         - alpha: L1 交叉熵损失的权重。
# #         - beta: L2 三元组损失的权重。
# #         - gamma: L3 覆盖损失的权重。
# #         """
# #         super(MatchingLoss, self).__init__()
# #         self.margin = margin
# #         self.alpha = alpha
# #         self.beta = beta
# #         self.gamma = gamma
# #         self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
# #         self.BCELOSS = torch.nn.BCELoss()

# #     def forward(self, matches, scores, features_a, features_b, boxes_a, boxes_b):
# #         """
# #         计算匹配损失。

# #         参数：
# #         - matches: List[Tensor], 每个样本的匹配索引，形状为 (N_i, 2)。
# #         - scores: List[Tensor], 每个样本的匹配置信度分数，形状为 (N_i,)。
# #         - features_a: List[Tensor], 每个样本的第一个图像的关键点特征，形状为 (M_i, D)。
# #         - features_b: List[Tensor], 每个样本的第二个图像的关键点特征，形状为 (K_i, D)。
# #         - boxes_a: List[Tensor], 每个样本的第一个图像的目标框信息，形状为 (M_i, 2)。
# #         - boxes_b: List[Tensor], 每个样本的第二个图像的目标框信息，形状为 (K_i, 2)。

# #         返回：
# #         - loss_total: 总损失。
# #         """
# #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         batch_losses = []

# #         for b in range(len(matches)):  # 遍历每个样本
# #             match_indices = matches[b]  # (N_i, 2)
# #             score = scores[b]  # (N_i,)
# #             feature_a = features_a[b].to(device)  # (M_i, D)
# #             feature_b = features_b[b].to(device)  # (K_i, D)  # (K_i, 2)
# #             box_a = boxes_a[b].to(device)  # 转换为张量并移动到设备
# #             box_b = boxes_b[b].to(device)

# #             # ========== 修正特征数量 ==========
# #             # feature_a 多于 box_a → 裁掉尾部 padding
# #             if feature_a.shape[0] > box_a.shape[0]:
# #                 feature_a = feature_a[: box_a.shape[0]]

# #             # feature_b 同理
# #             if feature_b.shape[0] > box_b.shape[0]:
# #                 feature_b = feature_b[: box_b.shape[0]]
# #             if len(match_indices) == 0:  # 如果没有匹配，跳过该样本
# #                 continue
            
# #             # 提取匹配的类别和 ID
# #             idx_a, idx_b = match_indices[:, 0], match_indices[:, 1]

# #             #过滤
# #             idx_a, idx_b = match_indices[:, 0], match_indices[:, 1]
# #             valid_mask = (idx_a >= 0) & (idx_a < box_a.shape[0]) & (idx_b >= 0) & (idx_b < box_b.shape[0])
# #             idx_a = idx_a[valid_mask].to(box_a.device)
# #             idx_b = idx_b[valid_mask].to(box_b.device)
# #             score = score[valid_mask].to(device) 

# # #########################################################
# #             class_a = box_a[idx_a, 0]  # (N_i,)
# #             id_a = box_a[idx_a, 1]  # (N_i,)
# #             class_b = box_b[idx_b, 0]
# #             id_b = box_b[idx_b, 1]  # (N_i,)

# #             # 计算匹配标签
# #             labels = ((class_a == class_b) & (id_a == id_b)).float().to(device)  # (N_i,)

# #             # === L1: 交叉熵损失 ===
# #             score = score.to(device)
# #             loss_id = self.BCELOSS(score, labels)

# #             # === L2: 三元组损失 ===
# #             # === L2: 三元组损失 ===
# #             feat_a = feature_a[idx_a]  # (N_i, D)
# #             feat_b = feature_b[idx_b]  # (N_i, D)

# #             anchors = []
# #             positives = []
# #             negatives = []

# #             for ii in range(len(idx_a)):
# #                 a_id = idx_a[ii]
# #                 b_id = idx_b[ii]

# #                 anchor = feature_a[a_id]
# #                 bid = box_b[b_id, 1]   # b 的 ID
# #                 aid = box_a[a_id, 1]   # a 的 ID

# #                 # -----------------------------
# #                 # 正确匹配：labels[ii]==1
# #                 # -----------------------------
# #                 if labels[ii] == 1:
# #                     positive = feature_b[b_id]

# #                     # negative = 任意 ID != aid 的样本
# #                     mask_neg = (box_b[:,1] != aid)
# #                     neg_candidates = feature_b[mask_neg]

# #                     if len(neg_candidates)==0:
# #                         continue

# #                     # 随机选择或 hardest negative
# #                     rand_idx = torch.randint(0, len(neg_candidates), (1,))
# #                     negative = neg_candidates[rand_idx]

# #                     anchors.append(anchor)
# #                     positives.append(positive)
# #                     negatives.append(negative)

# #                 # -----------------------------
# #                 # 错误匹配：labels[ii]==0
# #                 # -----------------------------
# #                 else:
# #                     negative = feature_b[b_id]

# #                     # positive = 右图中所有 ID == aid 的
# #                     mask_pos = (box_b[:,1] == aid)
# #                     pos_candidates = feature_b[mask_pos]

# #                     if len(pos_candidates)==0:
# #                         # 无真实正样本，跳过
# #                         continue

# #                     # 随机选择或 hardest positive
# #                     rand_idx = torch.randint(0, len(pos_candidates), (1,))
# #                     positive = pos_candidates[rand_idx]

# #                     anchors.append(anchor)
# #                     positives.append(positive)
# #                     negatives.append(negative)

# #             # === Triplet Loss ===
# #             if len(anchors) > 0:
# #                 A = torch.stack([a.reshape(-1) for a in anchors])   # shape (T,2048)
# #                 P = torch.stack([p.reshape(-1) for p in positives])
# #                 N = torch.stack([n.reshape(-1) for n in negatives])
# #                 loss_triplet = self.triplet_loss(A, P, N)
# #             else:
# #                 loss_triplet = torch.tensor(0.0, device=device)


# #             # # 采样负样本（随机选一个不同 ID 的框作为 negative）
# #             # negative_indices = torch.randint(0, feature_b.shape[0], (len(idx_a),), device=device)
# #             # negative_features = feature_b[negative_indices]  # (N_i, D)

# #             # loss_triplet = self.triplet_loss(feat_a, feat_b, negative_features)

# #             # === L3: 覆盖损失 ===
# #             # 惩罚未匹配的目标框
# #             matched_a = torch.zeros(len(box_a), device=device)
# #             matched_b = torch.zeros(len(box_b), device=device)
# #             # matched_a[idx_a] = 1  # 标记已匹配
# #             # matched_b[idx_b] = 1  # 标记已匹配
# #             # loss_unmatched = (1 - matched_a).sum() + (1 - matched_b).sum()

# #             loss_unmatched = len(matched_a)-len(idx_a)+len(matched_b)-len(idx_b)

# #             # 惩罚低分匹配 (鼓励高分匹配)
# #             loss_confidence = -torch.log(score + 1e-6).mean()  # 避免 log(0)

# #             loss_cover = 0.02 * loss_unmatched + loss_confidence

# #             # === 总损失 ===
# #             # self.beta=0
# #             # print(f"身份损失：{loss_triplet:>8.3f}, 三元组损失：{loss_triplet:>8.3f}, 惩罚损失：{self.gamma * loss_cover:>8.3f}")

# #             loss_total = self.alpha * loss_id + self.beta * loss_triplet + self.gamma * loss_cover
# #             batch_losses.append(loss_total)

# #         # 如果所有样本都没有有效匹配，返回 0 损失
# #         if len(batch_losses) == 0:
# #             return torch.tensor(0.0, device=device, requires_grad=True)

# #         # 对所有样本的损失求平均
# #         total_loss = torch.mean(torch.stack(batch_losses))
# #         return total_loss
# ##################修改版本1，正样本只有

# # class MatchingLoss(nn.Module):
# #     def __init__(self, margin=1, alpha=0.6, beta=0.3, gamma=0.1):#margin默认1
# #         """
# #         MatchingLoss 计算三部分损失：
# #         1. 交叉熵损失 (L1) - 计算匹配目标框的类别和 ID 误差。
# #         2. 三元组损失 (L2) - 计算匹配目标框的特征向量间的距离。
# #         3. 覆盖损失 (L3) - 让更多目标框被匹配，匹配分数更高。

# #         参数：
# #         - margin: Triplet loss 的 margin 超参数。
# #         - alpha: L1 交叉熵损失的权重。
# #         - beta: L2 三元组损失的权重。
# #         - gamma: L3 覆盖损失的权重。
# #         """
# #         super(MatchingLoss, self).__init__()
# #         self.margin = margin
# #         self.alpha = alpha
# #         self.beta = beta
# #         self.gamma = gamma
# #         self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
# #         self.BCELOSS = torch.nn.BCELoss()

# #     def forward(self, matches, scores, features_a, features_b, boxes_a, boxes_b):
# #         """
# #         计算匹配损失。

# #         参数：
# #         - matches: List[Tensor], 每个样本的匹配索引，形状为 (N_i, 2)。
# #         - scores: List[Tensor], 每个样本的匹配置信度分数，形状为 (N_i,)。
# #         - features_a: List[Tensor], 每个样本的第一个图像的关键点特征，形状为 (M_i, D)。
# #         - features_b: List[Tensor], 每个样本的第二个图像的关键点特征，形状为 (K_i, D)。
# #         - boxes_a: List[Tensor], 每个样本的第一个图像的目标框信息，形状为 (M_i, 2)。
# #         - boxes_b: List[Tensor], 每个样本的第二个图像的目标框信息，形状为 (K_i, 2)。

# #         返回：
# #         - loss_total: 总损失。
# #         """
# #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         batch_losses = []

# #         for b in range(len(matches)):  # 遍历每个样本
# #             match_indices = matches[b]  # (N_i, 2)
# #             score = scores[b]  # (N_i,)
# #             feature_a = features_a[b].to(device)  # (M_i, D)
# #             feature_b = features_b[b].to(device)  # (K_i, D)  # (K_i, 2)
# #             box_a = boxes_a[b].to(device)  # 转换为张量并移动到设备
# #             box_b = boxes_b[b].to(device)

# #             # ========== 修正特征数量 ==========
# #             # feature_a 多于 box_a → 裁掉尾部 padding
# #             if feature_a.shape[0] > box_a.shape[0]:
# #                 feature_a = feature_a[: box_a.shape[0]]

# #             # feature_b 同理
# #             if feature_b.shape[0] > box_b.shape[0]:
# #                 feature_b = feature_b[: box_b.shape[0]]
# #             if len(match_indices) == 0:  # 如果没有匹配，跳过该样本
# #                 continue
            
# #             # 提取匹配的类别和 ID
# #             idx_a, idx_b = match_indices[:, 0], match_indices[:, 1]

# #             #过滤
# #             idx_a, idx_b = match_indices[:, 0], match_indices[:, 1]
# #             valid_mask = (idx_a >= 0) & (idx_a < box_a.shape[0]) & (idx_b >= 0) & (idx_b < box_b.shape[0])
# #             idx_a = idx_a[valid_mask].to(box_a.device)
# #             idx_b = idx_b[valid_mask].to(box_b.device)
# #             score = score[valid_mask].to(device) 

# # #########################################################
# #             class_a = box_a[idx_a, 0]  # (N_i,)
# #             id_a = box_a[idx_a, 1]  # (N_i,)
# #             class_b = box_b[idx_b, 0]
# #             id_b = box_b[idx_b, 1]  # (N_i,)

# #             # 计算匹配标签
# #             labels = ((class_a == class_b) & (id_a == id_b)).float().to(device)  # (N_i,)

# #             # === L1: 交叉熵损失 ===
# #             score = score.to(device)
# #             loss_id = self.BCELOSS(score, labels)

# #             # === L2: 三元组损失 ===
# #             feat_a = feature_a[idx_a]  # (N_i, D)
# #             feat_b = feature_b[idx_b]  # (N_i, D)
# #             # === L2: Triplet Loss ===

# #             # 找出真实正样本（positive）
# #             # labels = 1 表示 class 和 id 都一致
# #             positive_mask = (labels == 1)

# #             if positive_mask.sum() > 0:
# #                 # 真实正样本的 anchor/positive
# #                 feat_a_pos = feat_a[positive_mask]   # (P, D)
# #                 feat_b_pos = feat_b[positive_mask]   # (P, D)

# #                 # ---- 构建真实负样本 negative ----
# #                 # negative = 右图中 id != 当前 anchor id 的所有特征

# #                 id_pos = idx_a[positive_mask]   # 正样本的 id 序列 (P,)
# #                 negative_features_list = []

# #                 for pid in id_pos:
# #                     # 右图中所有 id != pid 的目标，都可以作为 negative
# #                     mask_neg = (box_b[:, 1] != pid)    # (K_i,)
# #                     neg_candidates = feature_b[mask_neg]

# #                     if len(neg_candidates) == 0:
# #                         # 如果没有负样本（极少情况），随机选一个
# #                         rand_idx = torch.randint(0, feature_b.shape[0], (1,))
# #                         neg = feature_b[rand_idx]
# #                     else:
# #                         # 随机从“不同ID的对象”里选一个 hard negative
# #                         rand_idx = torch.randint(0, len(neg_candidates), (1,))
# #                         neg = neg_candidates[rand_idx]

# #                     negative_features_list.append(neg)

# #                 negative_features_pos = torch.cat(negative_features_list, dim=0)  # (P, D)

# #                 # ---- 真正的 Triplet Loss ----
# #                 loss_triplet = self.triplet_loss(
# #                     feat_a_pos,
# #                     feat_b_pos,
# #                     negative_features_pos
# #                 )
# #             else:
# #                 # 如果没有正样本，此 batch 不计算 triplet
# #                 loss_triplet = torch.tensor(0.0, device=device)

# #             # # 采样负样本（随机选一个不同 ID 的框作为 negative）
# #             # negative_indices = torch.randint(0, feature_b.shape[0], (len(idx_a),), device=device)
# #             # negative_features = feature_b[negative_indices]  # (N_i, D)

# #             # loss_triplet = self.triplet_loss(feat_a, feat_b, negative_features)

# #             # === L3: 覆盖损失 ===
# #             # 惩罚未匹配的目标框
# #             matched_a = torch.zeros(len(box_a), device=device)
# #             matched_b = torch.zeros(len(box_b), device=device)
# #             # matched_a[idx_a] = 1  # 标记已匹配
# #             # matched_b[idx_b] = 1  # 标记已匹配
# #             # loss_unmatched = (1 - matched_a).sum() + (1 - matched_b).sum()

# #             loss_unmatched = len(matched_a)-len(idx_a)+len(matched_b)-len(idx_b)

# #             # 惩罚低分匹配 (鼓励高分匹配)
# #             loss_confidence = -torch.log(score + 1e-6).mean()  # 避免 log(0)

# #             loss_cover = 0.02 * loss_unmatched + loss_confidence

# #             # === 总损失 ===
# #             # self.beta=0
# #             # print(f"身份损失：{loss_triplet:>8.3f}, 三元组损失：{loss_triplet:>8.3f}, 惩罚损失：{self.gamma * loss_cover:>8.3f}")

# #             loss_total = self.alpha * loss_id + self.beta * loss_triplet + self.gamma * loss_cover
# #             batch_losses.append(loss_total)

# #         # 如果所有样本都没有有效匹配，返回 0 损失
# #         if len(batch_losses) == 0:
# #             return torch.tensor(0.0, device=device, requires_grad=True)

# #         # 对所有样本的损失求平均
# #         total_loss = torch.mean(torch.stack(batch_losses))
# #         return total_loss
    
# # from modulefinder import Module
# # import torch
# # import torch.nn as nn




class MatchingLoss(nn.Module):
    def __init__(self, margin=1, alpha=0.6, beta=0.3, gamma=0.1):
        """
        MatchingLoss 计算三部分损失：
        1. 交叉熵损失 (L1) - 计算匹配目标框的类别和 ID 误差。
        2. 三元组损失 (L2) - 计算匹配目标框的特征向量间的距离。
        3. 覆盖损失 (L3) - 让更多目标框被匹配，匹配分数更高。

        参数：
        - margin: Triplet loss 的 margin 超参数。
        - alpha: L1 交叉熵损失的权重。
        - beta: L2 三元组损失的权重。
        - gamma: L3 覆盖损失的权重。
        """
        super(MatchingLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        self.BCELOSS = torch.nn.BCELoss()

    def forward(self, matches, scores, features_a, features_b, boxes_a, boxes_b):
        """
        计算匹配损失。

        参数：
        - matches: List[Tensor], 每个样本的匹配索引，形状为 (N_i, 2)。
        - scores: List[Tensor], 每个样本的匹配置信度分数，形状为 (N_i,)。
        - features_a: List[Tensor], 每个样本的第一个图像的关键点特征，形状为 (M_i, D)。
        - features_b: List[Tensor], 每个样本的第二个图像的关键点特征，形状为 (K_i, D)。
        - boxes_a: List[Tensor], 每个样本的第一个图像的目标框信息，形状为 (M_i, 2)。
        - boxes_b: List[Tensor], 每个样本的第二个图像的目标框信息，形状为 (K_i, 2)。

        返回：
        - loss_total: 总损失。
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_losses = []

        for b in range(len(matches)):  # 遍历每个样本
            match_indices = matches[b]  # (N_i, 2)
            score = scores[b]  # (N_i,)
            feature_a = features_a[b].to(device)  # (M_i, D)
            feature_b = features_b[b].to(device)  # (K_i, D)  # (K_i, 2)
            box_a = boxes_a[b].to(device)  # 转换为张量并移动到设备
            box_b = boxes_b[b].to(device)
            if len(match_indices) == 0:  # 如果没有匹配，跳过该样本
                continue
            
            # 提取匹配的类别和 ID
            idx_a, idx_b = match_indices[:, 0], match_indices[:, 1]

            #过滤
            idx_a, idx_b = match_indices[:, 0], match_indices[:, 1]
            valid_mask = (idx_a >= 0) & (idx_a < box_a.shape[0]) & (idx_b >= 0) & (idx_b < box_b.shape[0])
            idx_a = idx_a[valid_mask].to(box_a.device)
            idx_b = idx_b[valid_mask].to(box_b.device)
            score = score[valid_mask].to(device) 

#########################################################
            class_a = box_a[idx_a, 0]  # (N_i,)
            id_a = box_a[idx_a, 1]  # (N_i,)
            class_b = box_b[idx_b, 0]
            id_b = box_b[idx_b, 1]  # (N_i,)

            # 计算匹配标签
            labels = ((class_a == class_b) & (id_a == id_b)).float().to(device)  # (N_i,)

            # === L1: 交叉熵损失 ===
            score = score.to(device)
            loss_id = self.BCELOSS(score, labels)

            # === L2: 三元组损失 ===
            feat_a = feature_a[idx_a]  # (N_i, D)
            feat_b = feature_b[idx_b]  # (N_i, D)

            # 采样负样本（随机选一个不同 ID 的框作为 negative）
            negative_indices = torch.randint(0, feature_b.shape[0], (len(idx_a),), device=device)
            negative_features = feature_b[negative_indices]  # (N_i, D)

            loss_triplet = self.triplet_loss(feat_a, feat_b, negative_features)

            # === L3: 覆盖损失 ===
            # 惩罚未匹配的目标框
            matched_a = torch.zeros(len(box_a), device=device)
            matched_b = torch.zeros(len(box_b), device=device)
            # matched_a[idx_a] = 1  # 标记已匹配
            # matched_b[idx_b] = 1  # 标记已匹配
            # loss_unmatched = (1 - matched_a).sum() + (1 - matched_b).sum()

            loss_unmatched = len(matched_a)-len(idx_a)+len(matched_b)-len(idx_b)

            # 惩罚低分匹配 (鼓励高分匹配)
            loss_confidence = -torch.log(score + 1e-6).mean()  # 避免 log(0)

            loss_cover = 0.02 * loss_unmatched + loss_confidence

            # === 总损失 ===
            # self.beta=0
            loss_total = self.alpha * loss_id + self.beta * loss_triplet + self.gamma * loss_cover
            batch_losses.append(loss_total)

        # 如果所有样本都没有有效匹配，返回 0 损失
        if len(batch_losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 对所有样本的损失求平均
        total_loss = torch.mean(torch.stack(batch_losses))
        return total_loss
    

