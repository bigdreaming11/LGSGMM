import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # 动态进度条
import os
from datetime import datetime
from zz.dataloader.dataset import pad_and_mask
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # 动态进度条
import matplotlib.pyplot as plt
from datetime import datetime



def filter_matches_and_scores(matches, scores, boxes_a, boxes_b):
    """
    matches: List[Tensor]，每个元素 shape (n, 2)
    scores:  List[Tensor]，每个元素 shape (n,)
    boxes_a: List[Tensor]，每个元素 shape (m, 2)
    boxes_b: List[Tensor]，每个元素 shape (k, 2)
    """
    filtered_matches = []
    filtered_scores = []

    for i in range(len(matches)):
        match = matches[i]   # (n, 2)
        score = scores[i]    # (n,)
        box_a = boxes_a[i]
        box_b = boxes_b[i]

        if match.numel() == 0:  # 空的，直接跳过
            filtered_matches.append(match)
            filtered_scores.append(score)
            continue

        # 拆分索引
        idx_a = match[:, 0]
        idx_b = match[:, 1]

        # 合法性检查
        valid_mask = (idx_a >= 0) & (idx_a < box_a.shape[0]) & \
                     (idx_b >= 0) & (idx_b < box_b.shape[0])

        # 应用过滤
        match = match[valid_mask]
        score = score[valid_mask]

        filtered_matches.append(match)
        filtered_scores.append(score)

    return filtered_matches, filtered_scores

class resnet_match_Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, num_epochs, scheduler,
                 save_dir="./checkpoints", pretrained_weights_path=None):
        """
        初始化训练器
        :param model: 模型实例
        :param criterion: 损失函数
        :param optimizer: 优化器
        :param train_loader: 训练数据加载器
        :param val_loader: 验证数据加载器（可选）
        :param num_epochs: 总训练轮数
        :param scheduler: 学习率调度器（可选）
        :param save_dir: 模型保存路径
        :param pretrained_weights_path: 预训练权重路径（可选）
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.save_dir = save_dir

        # 初始化损失记录列表
        self.train_losses = []
        self.val_losses = []
        self.train_MDA=[]
        self.val_MDA=[]

    def plot_and_save_loss(self, epoch):
        """
        绘制训练和验证损失曲线，并保存为图片。
        :param epoch: 当前 epoch 数
        """
        # 创建保存图像的文件夹
        loss_dir = os.path.join(self.save_dir, "loss_plots")
        os.makedirs(loss_dir, exist_ok=True)

        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Train Loss", color="blue")
        if self.val_loader is not None:
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss", color="orange")
        # plt.plot(range(1, len(self.train_MDA) + 1), self.train_MDA, label="Train MDA", color="green")
        plt.plot(range(1, len(self.val_MDA) + 1), self.val_MDA, label="val MDA", color="red")
        plt.title(f"Loss Curve (Epoch {epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 当前时间戳
        filename = f"loss_plot_{timestamp}_epoch_{epoch}.png"
        filepath = os.path.join(loss_dir, filename)
        plt.savefig(filepath)
        plt.close()

        print(f"Loss plot saved to {filepath}")

    def save_model(self, epoch):
        """
        保存模型，文件名为当前时间 + epoch，并保存损失曲线。
        :param epoch: 当前 epoch 数
        """
        # 保存模型
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 当前时间戳
        filename = f"model_{timestamp}_epoch_{epoch}.pth"
        filepath = os.path.join(self.save_dir, filename)

        if isinstance(self.model, nn.DataParallel):  # 如果使用 DataParallel
            torch.save(self.model.module.state_dict(), filepath)
        else:
            torch.save(self.model.state_dict(), filepath)

        print(f"Model saved to {filepath}")

        # 绘制并保存损失曲线
        self.plot_and_save_loss(epoch)

    def MDA(self,boxes0 ,boxes1,matches ):
        match_t=matches#tensor
        GA=0#真值匹配数
        for box0 in boxes0:
            for box1 in boxes1:
                if box0[0]==box1[0] and box0[1]==box1[1]:
                    GA+=1
        
        TA=0#正确匹配
        FA=0#错误匹配
        
        for match in match_t:
            id0=match[0]
            id1=match[1]
            if boxes0[id0][0]==boxes1[id1][0] and boxes0[id0][1]==boxes1[id1][1] :
                TA+=1
            else:
                FA+=1
        #统计漏匹配MA
        MA=GA-TA-FA
        if GA+FA+MA==0:
            scores=0
        else:
            scores=TA/(GA+FA+MA)
        # precision=TA/(FA+TA)
        # recall=(TA+FA)/GA
        return scores

    def train_epoch(self):
        """单个训练周期"""
        self.model.train()
        torch.set_grad_enabled(True)
        total_loss = 0.0
        num_samples = 0
        total_train_MDA = 0.0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Train",
            unit="batch",
            leave=True
        )

        for batch in progress_bar:
            images_a = batch["image_a"]
            images_b = batch["image_b"]
            boxes_a = batch["boxes_a"]
            boxes_b = batch["boxes_b"]
            keypoints_a = batch["keypoints_a"]
            keypoints_b = batch["keypoints_b"]

            max_len_a = max(len(patches) for patches in boxes_a)
            max_len_b = max(len(patches) for patches in boxes_b)

            padded_keypoints_a, mask_a = pad_and_mask(keypoints_a, max_len_a)
            padded_boxes_a, _ = pad_and_mask(boxes_a, max_len_a)

            padded_keypoints_b, mask_b = pad_and_mask(keypoints_b, max_len_b)
            padded_boxes_b, _ = pad_and_mask(boxes_b, max_len_b)

            images_a = images_a.to(device)
            padded_keypoints_a = padded_keypoints_a.to(device)
            mask_a = mask_a.to(device)
            padded_boxes_a = padded_boxes_a.to(device)

            images_b = images_b.to(device)
            padded_keypoints_b = padded_keypoints_b.to(device)
            mask_b = mask_b.to(device)
            padded_boxes_b = padded_boxes_b.to(device)

            points0, points1, matches, scores, features_a, features_b ,self_attn0,self_attn1,cross_attn01,cross_attn10 = self.model(
                images_a, images_b, padded_keypoints_a, padded_keypoints_b, mask_a, mask_b
            )

            boxes_a = [torch.tensor(b).to(device) for b in boxes_a]
            boxes_b = [torch.tensor(b).to(device) for b in boxes_b]
            ########确保matches有效##########################
            matches, scores=filter_matches_and_scores(matches, scores, boxes_a, boxes_b)
            loss = self.criterion(matches, scores, features_a, features_b, boxes_a, boxes_b)

            self.optimizer.zero_grad()
            loss.backward()
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient of {name}: {param.grad}")           
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_samples += 1

            current_loss = total_loss / num_samples
            # progress_bar.set_postfix({
            #     "Loss": f"{current_loss:.4f}",
            #     "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            # })
        # 计算当前批次的 MDA 平均值
            boxes0 = batch["boxes_a"]
            boxes1 = batch["boxes_b"]
            MDA_each = []
            for i in range(len(boxes0)):
                current_MDA = self.MDA(boxes0[i], boxes1[i], matches[i])
                MDA_each.append(current_MDA)

            batch_MDA = np.mean(MDA_each)  # 当前批次的 MDA 平均值
            total_train_MDA += batch_MDA  # 累积 MDA 值（乘以批次样本数）

            # 更新进度条显示
            avg_loss = total_loss / num_samples
            avg_MDA=total_train_MDA /num_samples
            progress_bar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Train MDA": f"{avg_MDA:.4f}",  # 显示当前批次的 MDA
                "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        # 计算整个训练周期的平均损失和 MDA
        avg_train_loss = total_loss / num_samples
        avg_train_MDA = total_train_MDA / num_samples  # 整个训练周期的平均 MDA

        # 记录训练损失和 MDA
        self.train_losses.append(avg_train_loss)
        # self.train_mda_scores.append(avg_train_MDA)  # 记录训练 MDA
        self.train_MDA.append(avg_train_MDA )
        return avg_train_loss, avg_train_MDA  # 返回平均损失和平均 MDA
    
    def validate_epoch(self):
        """单个验证周期"""
        self.model.eval()
        val_loss = 0.0
        val_samples = 0
        total_val_MDA =0.0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        progress_bar = tqdm(
            self.val_loader,
            desc="Validation",
            unit="batch",
            leave=True
        )

        with torch.no_grad():
            for batch in progress_bar:
                images_a = batch["image_a"]
                images_b = batch["image_b"]
                boxes_a = batch["boxes_a"]
                boxes_b = batch["boxes_b"]
                keypoints_a = batch["keypoints_a"]
                keypoints_b = batch["keypoints_b"]

                max_len_a = max(len(patches) for patches in boxes_a)
                max_len_b = max(len(patches) for patches in boxes_b)

                padded_keypoints_a, mask_a = pad_and_mask(keypoints_a, max_len_a)
                padded_boxes_a, _ = pad_and_mask(boxes_a, max_len_a)

                padded_keypoints_b, mask_b = pad_and_mask(keypoints_b, max_len_b)
                padded_boxes_b, _ = pad_and_mask(boxes_b, max_len_b)

                images_a = images_a.to(device)
                padded_keypoints_a = padded_keypoints_a.to(device)
                mask_a = mask_a.to(device)
                padded_boxes_a = padded_boxes_a.to(device)

                images_b = images_b.to(device)
                padded_keypoints_b = padded_keypoints_b.to(device)
                mask_b = mask_b.to(device)
                padded_boxes_b = padded_boxes_b.to(device)

                points0, points1, matches, scores, features_a, features_b ,self_attn0,self_attn1,cross_attn01,cross_attn10= self.model(
                    images_a, images_b, padded_keypoints_a, padded_keypoints_b, mask_a, mask_b
                )
                
                boxes_a = [torch.tensor(b).to(device) for b in boxes_a]
                boxes_b = [torch.tensor(b).to(device) for b in boxes_b]
                matches, scores=filter_matches_and_scores(matches, scores, boxes_a, boxes_b)
                loss = self.criterion(matches, scores, features_a, features_b, padded_boxes_a, padded_boxes_b)

                batch_size = len(boxes_a)
                val_loss += loss.item() * batch_size
                val_samples += batch_size



                current_loss = val_loss / val_samples
                # progress_bar.set_postfix({
                #     "Loss": f"{current_loss:.4f}"
                # })

                
                boxes0=batch["boxes_a"]
                boxes1=batch["boxes_b"]
                MDA_each=[]
                for i in range(len(boxes0)):
                    current_MDA=self.MDA(boxes0[i] ,boxes1[i],matches[i] )
                    MDA_each.append(current_MDA)
              
                val_MDA = np.mean(MDA_each)

                total_val_MDA += val_MDA * len(images_a)          
                progress_bar.set_postfix({
                    "Loss": f"{current_loss:.4f}",
                    "MDA": f"{val_MDA:.4f}",
                })
                 # 当前批次的 MDA 平均值


        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0.0
        self.val_losses.append(avg_val_loss)  # 记录验证损失
        avg_val_MDA = total_val_MDA  /val_samples  
        self.val_MDA.append(avg_val_MDA )

        return avg_val_loss,avg_val_MDA

    def train_and_validate(self):
        """训练和验证主循环"""
        for epoch in range(self.num_epochs):
            avg_train_loss, avg_train_MDA = self.train_epoch()
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} completed. Avg Train Loss: {avg_train_loss:.4f}. Avg train MDA:{avg_train_MDA:.4f}")

            if self.val_loader is not None:
                avg_val_loss,avg_val_MDA = self.validate_epoch()
                print(f"Validation Epoch {epoch + 1}/{self.num_epochs} completed. Avg Val Loss: {avg_val_loss:.4f}. Avg val MDA:{avg_val_MDA:.4f}")

                if self.scheduler is not None:
                    self.scheduler.step(avg_val_loss)

            # 每隔 10 个 epoch 保存一次模型和损失曲线
            if (epoch + 1) % 2== 0:
                self.save_model(epoch + 1)

        # 最终保存模型和损失曲线
        self.save_model(self.num_epochs)