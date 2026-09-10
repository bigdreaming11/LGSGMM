import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

import torch
import random
import numpy as np
from zz.dataloader.dataset import TargetConnectionDataset
from zz.all_nn.backbone_match import TargetConnectionModel
from zz.train.trainer import resnet_match_Trainer
from zz.train.loss import MatchingLoss#############################################这里修改loss类型
from torch.nn import DataParallel
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from zz.dataloader.dataset import custom_collate_fn,CustomDataParallel

##############
# 设置随机种子#
##############
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##############
# 打印显卡信息#
##############
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    print(f"  GPU {i}: {gpu_name}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    set_seed(42)  # 确保结果可复现

    # 预训练模型路径
    path_pretrained = r"C:\Users\Lenovo\Desktop\code\250321\LightGlue-main\weights\aa.pth"

    # txt_a_train = r"/mnt/8t/zz/zz/data/tianda_gap20/train/txt_a"
    # txt_b_train = r"/mnt/8t/zz/zz/data/tianda_gap20/train/txt_b"
    # img_a_train = r"/mnt/8t/zz/zz/data/tianda_gap20/train/img_a"
    # img_b_train = r"/mnt/8t/zz/zz/data/tianda_gap20/train/img_b"

    # txt_a_val = r"/mnt/8t/zz/zz/data/tianda_gap20/val/txt_a"
    # txt_b_val = r"/mnt/8t/zz/zz/data/tianda_gap20/val/txt_b"
    # img_a_val = r"/mnt/8t/zz/zz/data/tianda_gap20/val/img_a"
    # img_b_val = r"/mnt/8t/zz/zz/data/tianda_gap20/val/img_b"



    txt_a_train = r"/mnt/8t/zz/zz/data/tianda_zhengshi_gap10/train/txt_a"
    txt_b_train = r"/mnt/8t/zz/zz/data/tianda_zhengshi_gap10/train/txt_b"
    img_a_train = r"/mnt/8t/zz/zz/data/tianda_zhengshi_gap10/train/img_a"
    img_b_train = r"/mnt/8t/zz/zz/data/tianda_zhengshi_gap10/train/img_b"

    txt_a_val = r"/mnt/8t/zz/zz/data/tianda_zhengshi_gap10/val/txt_a"
    txt_b_val = r"/mnt/8t/zz/zz/data/tianda_zhengshi_gap10/val/txt_b"
    img_a_val = r"/mnt/8t/zz/zz/data/tianda_zhengshi_gap10/val/img_a"
    img_b_val = r"/mnt/8t/zz/zz/data/tianda_zhengshi_gap10/val/img_b"

    lightglue_pretrain="/mnt/8t/zz/zz/glue0529/weights/superpoint_lightglue.pth"    

    train_path = [img_a_train, img_b_train, txt_a_train, txt_b_train]
    val_path = [img_a_val, img_b_val, txt_a_val, txt_b_val]

    # SuperPoint 和 LightGlue 的配置
    filter_threshold = 0
    # 训练参数
    batch_size = 4
    num_epochs =200
    learning_rate = 0.0001


    # 数据集和数据加载器
    train_dataset = TargetConnectionDataset(train_path)
    val_dataset = TargetConnectionDataset(val_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn

    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    # 模型初始化

    model = TargetConnectionModel(
        backbone_out_dim=2048,
        filter_threshold=filter_threshold,
        lightglue_pretrain=lightglue_pretrain
    ).to(device)

    if num_gpus > 1:
        # model = DataParallel(model)
        model=CustomDataParallel(model)

    model = model.to(device)

    # 损失函数、优化器和学习率调度器
    criterion = MatchingLoss ().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    # 初始化 Trainer
    trainer = resnet_match_Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        scheduler =scheduler,
        # save_dir=r"C:\Users\zhangzhao\Desktop\code\250330\LightGlue-main\LightGlue-main\LightGlue-main\out_pth"
        save_dir=r"/mnt/8t/zz/zz/glue0529/out_pth/bt4"
    )

    # 开始训练和验证
    trainer.train_and_validate()

# 无预训练权重，单卡训练，修改loss