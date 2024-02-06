import torch

a1 = torch.rand(4, 3, 16, 32)
a2 = torch.rand(4, 3, 16, 32)
# torch.Size([4, 3, 32, 32])
print(torch.cat([a1, a2], dim=2).shape)
# 往2索引位置插入一个[2]维度的向量，原本2、3索引向后移一位变为3、4
# torch.Size([4, 3, 2, 16, 32])
print(torch.stack([a1, a2], dim=2).shape)
a = torch.rand(32, 8)
b = torch.rand(32, 8)
# 往0索引位置插入一个[2]维度的向量，原本0、1索引向后移一位变为1、2
# torch.Size([2, 32, 8])
print(torch.stack([a, b], dim=0).shape)
# Cat VS stack
a = torch.rand(32, 8)
b = torch.rand(30, 8)
# torch.Size([32, 8])
print(a.shape)
# torch.Size([30, 8])
print(b.shape)
# RuntimeError: stack expects each tensor to be equal size, but got [32, 8] at entry 0 and [30, 8] at entry 1
# torch.stack([a,b], dim=0)
# torch.Size([62, 8])
print(torch.cat([a,b], dim=0).shape)