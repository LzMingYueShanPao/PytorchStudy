import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(1, 28*28)
# torch.Size([1, 784])
print(x.shape)

"""
nn.Linear是PyTorch深度学习框架中的一个类，用于定义一个线性变换（linear transformation）模块。它可以被用来构建神经网络的层。
该类的构造函数如下：
    nn.Linear(in_features, out_features, bias=True)
参数说明：
    in_features：输入特征的数量。
    out_features：输出特征的数量。
    bias：是否使用偏置项。默认为True，表示使用偏置项；设置为False则不使用。
"""
layer1 = nn.Linear(784, 200)
layer2 = nn.Linear(200, 200)
layer3 = nn.Linear(200, 10)

# [1, 784] x [784, 200] = [1, 200]
x = layer1(x)
# torch.Size([1, 200])
print(x.shape)
"""
F.relu()是PyTorch深度学习框架中的一个函数，用于实现ReLU（Rectified Linear Unit）激活函数。ReLU是一种常用的非线性激活函数，通常被用于神经网络的隐藏层。
该函数的定义如下：
    F.relu(input, inplace=False)
参数说明：
    input：输入张量。
    inplace：是否进行原地操作。默认为False，表示不进行原地操作；设置为True，则会直接修改输入张量。
"""
x = F.relu(x, inplace=True)
# torch.Size([1, 200])
print(x.shape)
# [1, 200] x [200, 200] = [1, 200]
x = layer2(x)
# torch.Size([1, 200])
print(x.shape)
x = F.relu(x, inplace=True)
# torch.Size([1, 200])
print(x.shape)
# [1, 200] x [200, 10] = [1, 10]
x = layer3(x)
# torch.Size([1, 10])
print(x.shape)
x = F.relu(x, inplace=True)
# torch.Size([1, 10])
print(x.shape)