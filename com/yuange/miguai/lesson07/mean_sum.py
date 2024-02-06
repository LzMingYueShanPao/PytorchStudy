import torch

temp = torch.arange(8)
# tensor([0, 1, 2, 3, 4, 5, 6, 7])
print(temp)
temp2 = temp.view(2, 4)
# tensor([[0, 1, 2, 3],
#         [4, 5, 6, 7]])
print(temp2)
a = temp2.float()
# tensor([[0., 1., 2., 3.],
#         [4., 5., 6., 7.]])
print(a)
"""
torch.min() 是 PyTorch 中的函数之一，用于返回张量中的最小值。它可以在指定的维度上计算最小值，或者在整个张量中找到全局最小值。
torch.max() 是 PyTorch 中的函数之一，用于返回张量中的最大值。它可以在指定的维度上计算最大值，或者在整个张量中找到全局最大值。
torch.mean() 是 PyTorch 中的函数之一，用于计算张量在指定维度上的均值。它可以计算整个张量的均值，或者在指定的维度上计算均值。
torch.prod() 是 PyTorch 中的函数之一，用于计算张量在指定维度上的乘积。它可以计算整个张量的乘积，或者在指定的维度上计算乘积。
torch.sum() 是 PyTorch 中的函数之一，用于计算张量在指定维度上的和。它可以计算整个张量的和，或者在指定的维度上计算和。
torch.argmax() 是 PyTorch 中的函数之一，用于在张量中找到最大值所在的索引。它可以在整个张量或指定维度上找到最大值的索引。
torch.argmin() 函数用于找到张量中最小值所在的索引。它可以在整个张量或指定维度上找到最小值的索引。
"""
# tensor(0.) tensor(7.) tensor(3.5000) tensor(0.)
print(a.min(), a.max(), a.mean(), a.prod())
# tensor(28.)
print(a.sum())
# tensor(7) tensor(0)
print(a.argmax(), a.argmin())
a = a.view(1, 2, 4)
# tensor([[[0., 1., 2., 3.],
#          [4., 5., 6., 7.]]])
print(a)
# tensor(7) tensor(0)
print(a.argmax(), a.argmin())
a = torch.rand(2, 3, 4)
# tensor(23)
print(a.argmax())
a = torch.randn(4, 10)
# tensor([ 0.3285,  0.3024, -0.3104, -0.5542,  0.2108, -0.8404,  1.1314,  1.6674,
#         -0.8779, -0.2555])
print(a[0])
# tensor(7)
print(a.argmax())
# tensor([0, 2, 1, 1, 1, 1, 0, 0, 2, 0])
print(a.argmax(dim=0))
# tensor([6, 0, 5, 8])
print(a.argmax(dim=1))
# tensor([[ 1.3932,  0.4904,  1.1432,  0.1257, -0.3030, -0.7637,  1.1842, -0.6374,
#           0.6361, -1.8163],
#         [ 0.6725, -0.4102, -0.6991,  0.8503, -0.9660,  1.1221,  0.7370, -0.9380,
#          -0.2376,  0.8645],
#         [-0.8183,  1.0662, -0.0133,  0.0723,  1.3837, -1.0600,  1.3414,  1.3559,
#           2.0205, -1.0193],
#         [ 0.4818,  1.5451,  0.5324,  0.6870, -0.7033,  1.2474,  0.0872,  0.1811,
#           0.7504,  0.5893]])
print(a)
# torch.return_types.max(
# values=tensor([0.8286, 1.4628, 1.1285, 1.7311]),
# indices=tensor([1, 3, 0, 1]))
print(a.max(dim=1))
# tensor([1, 3, 0, 1])
print(a.argmax(dim=1))
# torch.return_types.max(
# values=tensor([[0.8286],
#         [1.4628],
#         [1.1285],
#         [1.7311]]),
# indices=tensor([[1],
#         [3],
#         [0],
#         [1]]))
print(a.max(dim=1, keepdim=True))
# tensor([[1],
#         [3],
#         [0],
#         [1]])
print(a.argmax(dim=1, keepdim=True))