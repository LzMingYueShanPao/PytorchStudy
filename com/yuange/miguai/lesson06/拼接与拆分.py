import torch

a1 = torch.rand(4, 3, 32, 32)
a2 = torch.rand(5, 3, 32, 32)
# torch.Size([9, 3, 32, 32])
print(torch.cat([a1, a2], dim=0).shape)
a2 = torch.rand(4, 1, 32, 32)
# RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 3 but got size 1 for tensor number 1 in the list.
# print(torch.cat([a1, a2], dim=0).shape)
# torch.Size([4, 4, 32, 32])
print(torch.cat([a1, a2], dim=1).shape)
a1 = torch.rand(4, 3, 16, 32)
a2 = torch.rand(4, 3, 16, 32)
# torch.Size([4, 3, 32, 32])
print(torch.cat([a1, a2], dim=2).shape)


print('----------------------------------------')
"""
Statistics about scores（分数统计）
[class1-4, students, scores]
[class5-9, students, scores]
"""
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
# torch.Size([9, 32, 8])
print(torch.cat([a,b],dim=0).shape)