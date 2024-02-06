import torch
import torch.nn as nn
import torch.nn.functional as F

layer = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
x = torch.rand(1, 1, 14, 14)

x_out = layer.forward(x)
# torch.Size([1, 16, 14, 14])
print(x_out.shape)

"""
创建一个二维最大池化层。这个函数将输入张量进行2x2的池化操作，并使用2的步幅来移动池化窗口。
在池化操作中，每个窗口内的最大值被提取出来，从而降低了特征图的尺寸，同时保留了最重要的特征。
这在卷积神经网络中通常用于减小特征图的尺寸，以减少模型的参数数量并加快计算速度。
"""
layer = nn.MaxPool2d(2, stride=2)
out = layer(x_out)
# torch.Size([1, 16, 7, 7])
print(out.shape)

out = F.avg_pool2d(x_out, 2, stride=2)
# torch.Size([1, 16, 7, 7])
print(out.shape)