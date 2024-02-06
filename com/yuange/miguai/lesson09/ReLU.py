import torch
from torch import nn
import torch.nn.functional as F

x = torch.rand(1, 16, 7, 7)
"""
ReLU的定义是 f(x) = max(0, x)，即将输入值 x 小于 0 的部分截断为 0，大于等于 0 的部分保持不变。这个操作能够帮助网络引入非线性，提升模型的表达能力。
inplace=True 表示 ReLU 操作将会原地进行，即直接在输入张量上进行修改而不创建新的张量。这样可以节省内存空间，但同时也会改变输入张量的值。
如果不指定 inplace=True，则 ReLU 操作会创建一个新的张量作为输出。
"""
layer = nn.ReLU(inplace=True)

out = layer(x)
# torch.Size([1, 16, 7, 7])
print(out.shape)

out = F.relu(x)
# torch.Size([1, 16, 7, 7])
print(out.shape)