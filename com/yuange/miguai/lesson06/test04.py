import torch

# torch.randperm(10) 函数将会创建一个长度为 10 的随机排列张量，其中包含 0 到 9 的整数，每个数字恰好出现一次，且顺序是随机的。
# tensor([5, 4, 7, 0, 8, 1, 2, 6, 9, 3])
print(torch.randperm(10))
# 创建一个形状为 2x3 的张量，其中的元素是均匀随机分布在 [0, 1) 区间内的随机数。
a = torch.rand(2, 3)
# 创建一个形状为 2x2 的张量，其中的元素是均匀随机分布在 [0, 1) 区间内的随机数。
b = torch.rand(2, 2)
# # torch.randperm(2) 函数将会创建一个长度为 2 的随机排列张量，其中包含 0 到 1 的整数，每个数字恰好出现一次，且顺序是随机的。
idx = torch.randperm(2)
# tensor([0, 1])
print(idx)
# tensor([1, 0])
print(idx)
# 使用打散的idx数据当做a、b张量的索引去访问
# tensor([[0.2368, 0.0482, 0.6387],
#         [0.5997, 0.2516, 0.4364]])
print(a[idx])
# tensor([[0.4076, 0.5612],
#         [0.1863, 0.7893]])
print(b[idx])
# tensor([[0.2368, 0.0482, 0.6387],
#         [0.5997, 0.2516, 0.4364]]) tensor([[0.4076, 0.5612],
#         [0.1863, 0.7893]])
print(a, b)


print('----------------------------------------------------------')
# 创建一个形状为 3x3 的张量，并且张量中的所有元素都是 1。
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])
print(torch.ones(3, 3))
# 创建一个形状为 3x3 的张量，并且张量中的所有元素都是 0。
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])
print(torch.zeros(3, 3))
# 创建一个形状为 3x4 的单位矩阵，单位矩阵是一个主对角线上的元素全为 1，其余元素全为 0 的矩阵。
# tensor([[1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.]])
print(torch.eye(3, 4))
# 创建一个形状为 3x3 的方阵，该方阵是一个主对角线上的元素全为 1，其余元素全为 0 的矩阵。
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
print(torch.eye(3))
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])
a = torch.zeros(3, 3)
# 创建一个和张量 a 具有相同形状的张量，并且其中的所有元素都是 1。
print(torch.ones_like(a))