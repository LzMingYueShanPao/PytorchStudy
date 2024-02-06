import torch

a = torch.randn(4, 10)
# tensor([[ 0.2912,  1.1822, -0.0974, -0.6414,  0.7345, -0.9356,  0.3451, -0.0540,
#           1.5192, -0.2148],
#         [ 0.2621,  1.3402, -0.8220, -0.3104,  0.7208,  0.4336,  0.5446, -0.2453,
#          -0.3221, -0.9273],
#         [-0.6056,  1.0659, -0.8228,  0.0135, -0.5837, -1.6353, -0.4095,  0.5581,
#          -0.2597, -1.1458],
#         [ 1.5257, -0.3315,  0.2512,  0.3154,  0.4271, -1.0807,  0.1627, -1.0244,
#          -1.2035,  1.8029]])
print(a)
"""
在 PyTorch 中，torch.topk() 函数用于找到张量中最大的 k 个元素及其对应的索引。它可以在整个张量或指定维度上找到最大的 k 个元素及其对应的索引。
设置largest=False表示最小的k个元素及其对应的索引。
"""
# torch.return_types.topk(
# values=tensor([[1.5192, 1.1822, 0.7345],
#         [1.3402, 0.7208, 0.5446],
#         [1.0659, 0.5581, 0.0135],
#         [1.8029, 1.5257, 0.4271]]),
# indices=tensor([[8, 1, 4],
#         [1, 4, 6],
#         [1, 7, 3],
#         [9, 0, 4]]))
print(a.topk(3, dim=1))
# torch.return_types.topk(
# values=tensor([[-0.9356, -0.6414, -0.2148],
#         [-0.9273, -0.8220, -0.3221],
#         [-1.6353, -1.1458, -0.8228],
#         [-1.2035, -1.0807, -1.0244]]),
# indices=tensor([[5, 3, 9],
#         [9, 2, 8],
#         [5, 9, 2],
#         [8, 5, 7]]))
print(a.topk(3, dim=1, largest=False))
"""
在 PyTorch 中，kthvalue() 是一个张量的方法，用于计算张量中第 k 小元素的值及其索引位置。
kthvalue() 方法的使用方式如下：
    values, indices = tensor.kthvalue(k, dim=None, keepdim=False)
其中，tensor 是要进行操作的张量，k 是要计算的第 k 小元素的位置，dim 是可选的维度参数，指定在哪个维度上计算第 k 小元素，默认为 None 表示在整个张量上计算，keepdim 是一个布尔值参数，用于指定计算后是否保留计算维度，默认为 False 表示不保留。
kthvalue() 方法将返回两个张量：第一个张量 values 是第 k 小元素的值，第二个张量 indices 是第 k 小元素的索引位置。
"""
# torch.return_types.kthvalue(
# values=tensor([0.7345, 0.5446, 0.0135, 0.4271]),
# indices=tensor([4, 6, 3, 4]))
print(a.kthvalue(8, dim=1))
# torch.return_types.kthvalue(
# values=tensor([-0.2148, -0.3221, -0.8228, -1.0244]),
# indices=tensor([9, 8, 2, 7]))
print(a.kthvalue(3))
# torch.return_types.kthvalue(
# values=tensor([-0.2148, -0.3221, -0.8228, -1.0244]),
# indices=tensor([9, 8, 2, 7]))
print(a.kthvalue(3, dim=1))