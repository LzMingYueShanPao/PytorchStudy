import torch
import torch.nn.functional as F

x = torch.rand(1, 16, 7, 7)
"""
对输入张量 x 进行插值操作，scale_factor=2 表示对输入张量进行两倍的上采样，在高度和宽度上都放大了两倍的输入张量 x。
而 mode='nearest' 表示使用最近邻插值方法。最近邻插值方法会简单地使用最接近目标位置的输入像素值来进行插值，不进行任何加权平均操作。
"""
out = F.interpolate(x, scale_factor=2, mode='nearest')
# rch.Size([1, 16, 14, 14])
print(out.shape)

out = F.interpolate(x, scale_factor=3, mode='nearest')
# torch.Size([1, 16, 21, 21])
print(out.shape)