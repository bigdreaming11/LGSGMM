import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights,resnet101, ResNet101_Weights,resnet18,ResNet18_Weights
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import ConnectionPatch


def plot_losses(self):
    """
    绘制训练损失和验证损失曲线，并实时更新
    """
    # 清除之前的绘图内容
    self.ax.clear()

    # 绘制训练损失和验证损失
    self.ax.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Train Loss", color="blue")
    if self.val_loader is not None:
        self.ax.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss", color="orange")

    # 设置图表标题、标签和图例
    self.ax.set_xlabel("Epoch")
    self.ax.set_ylabel("Loss")
    self.ax.set_title("Training and Validation Loss")
    self.ax.legend()
    self.ax.grid(True)

    # 强制刷新图表
    plt.pause(0.01)


def visual_boxes(img0, img1, boxes0, boxes1):
    boxes1 = [box[:] for box in boxes1]
    box_color=(0,255,0)
    line_color=(0,255,0)
    thickness=2
    combined_img=cv2.hconcat([img0, img1])
    #shape输出height，weigh
    img2_offset=img0.shape[1]
    for box in boxes1:
        box[2]+=img2_offset
    for box in boxes0:
        class_id,id,x,y,w,h=box
        x1,y1=int(x-w/2),int(y-h/2)
        x2,y2=int(x+w/2),int(y+h/2)
        cv2.rectangle(combined_img,(x1,y1),(x2,y2),box_color,thickness)
    for box in boxes1:
        class_id,id,x,y,w,h=box
        x1,y1=int(x-w/2),int(y-h/2)
        x2,y2=int(x+w/2),int(y+h/2)
        cv2.rectangle(combined_img,(x1,y1),(x2,y2),box_color,thickness)
    for box0 in boxes0:
        for box1 in boxes1:
            if box0[0]==box1[0] and box0[1]==box1[1]:
                center0_x,center0_y=int(box0[2]),int(box0[3])
                center1_x,center1_y=int(box1[2]),int(box1[3])
                cv2.line(combined_img, (center0_x, center0_y), (center1_x, center1_y), line_color, thickness)

    return combined_img
#

def load_image_rgb(path):
    return Image.open(path).convert('RGB')
#
def load_image_rgb(rgb_list):
    # 将嵌套列表转换为 NumPy 数组
    rgb_array = np.array(rgb_list, dtype=np.uint8)

    # 验证数组形状是否为 (height, width, 3)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("输入的列表必须是 RGB 格式 (高度, 宽度, 3)")

    # 转换为 PIL 图像并返回
    return Image.fromarray(rgb_array)



def move_to_cuda(data_dict, device="cuda"):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in data_dict.items()
    }
