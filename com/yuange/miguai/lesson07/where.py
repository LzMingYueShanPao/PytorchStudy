import torch

"""
在PyTorch中，torch.where函数用于根据给定的条件选择元素。它的语法如下：
    torch.where(condition, x, y)
其中，condition是一个布尔张量，x和y是两个具有相同形状的张量。torch.where函数将返回一个新的张量，该张量中的元素来自于x和y，根据condition中对应位置的值来选择。
具体而言，当condition中的元素为True时，torch.where会选择x中对应位置的元素；当condition中的元素为False时，torch.where会选择y中对应位置的元素。
"""
cond = torch.rand(2,2)
# tensor([[0.2037, 0.3482],
#         [0.7595, 0.4757]])
print(cond)
a = torch.ones(2,2)
# tensor([[1., 1.],
#         [1., 1.]])
print(a)
b = torch.zeros(2,2)
# tensor([[0., 0.],
#         [0., 0.]])
print(b)
# tensor([[0., 0.],
#         [1., 0.]])
print(torch.where(cond > 0.5, a, b))