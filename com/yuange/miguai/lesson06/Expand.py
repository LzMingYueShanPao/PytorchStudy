import torch

a = torch.rand(4, 3, 32, 32)
# torch.Size([4, 3, 32, 32])
print(a.shape)
"""
transpose() 是 PyTorch 张量的一个方法，用于交换张量的维度。它可以用于转置矩阵、改变卷积核的通道顺序等操作。
transpose() 方法的使用方式如下：
    new_tensor = tensor.transpose(dim0, dim1)
其中，tensor 是要进行维度交换的原始张量，dim0 和 dim1 是要交换的维度的索引。
以下是 transpose() 方法的一些常见用法和注意事项：
    transpose() 方法会返回一个新的张量，与原始张量不共享内存。
    如果只是需要翻转张量的维度顺序，可以使用 torch.flip() 或者 torch.flipud()、torch.fliplr() 等函数。
"""
temp = a.transpose(1,3)
# torch.Size([4, 32, 32, 3])
print(temp.shape)
# RuntimeError: view size is not compatible with input tensor's size
# and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
# print(temp.view(4, 3 * 32 * 32).shape)
"""
在 PyTorch 中，contiguous() 是一个方法，用于确保张量的内存布局是连续的。
在内存中，张量的元素通常是按照一定的顺序存储的。然而，某些操作（如转置、切片等）可能会改变张量的内存布局，使其不再连续。这在某些情况下可能会导致性能下降或错误。
contiguous() 方法可以检查并返回一个连续的张量。如果原始张量已经是连续的，将直接返回原始张量；否则，将创建一个新的连续张量，并复制原始张量的数据。
以下是 contiguous() 方法的使用方式：
    contiguous_tensor = tensor.contiguous()
其中，tensor 是要检查连续性的原始张量，contiguous_tensor 是返回的连续张量。
"""
temp2 = temp.contiguous()
# torch.Size([4, 32, 32, 3])
print(temp2.shape)
temp3 = temp2.view(4, 3*32*32)
# torch.Size([4, 3072])
print(temp3.shape)
temp4 = temp3.view(4, 3, 32, 32)
# torch.Size([4, 3, 32, 32])
print(temp4.shape)
temp5 = temp3.view(4, 32, 32, 3)
# torch.Size([4, 32, 32, 3])
print(temp5.shape)
temp6 = temp5.transpose(0,1)
# torch.Size([32, 4, 32, 3])
print(temp6.shape)
temp7 = temp5.transpose(1,3)
# torch.Size([4, 3, 32, 32])
print(temp7.shape)


print('---------------------------------------------------')
b = torch.rand(32)
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
# torch.Size([1, 32, 1, 1])
print(b.shape)
# RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 4D
# b.t()
a = torch.randn(3, 4)
# torch.Size([3, 4])
print(a.shape)
# tensor([[ 0.9996, -0.8327, -1.2617, -0.1957],
#         [ 0.2665, -0.6010,  0.2669, -0.7235],
#         [ 0.0794, -0.5883,  0.5026, -1.8267]])
print(a)
# tensor([[ 0.9996,  0.2665,  0.0794],
#         [-0.8327, -0.6010, -0.5883],
#         [-1.2617,  0.2669,  0.5026],
#         [-0.1957, -0.7235, -1.8267]])
print(a.t())

print('---------------------------------------------------')
# torch.Size([4, 1024, 1, 1])
print(b.repeat(4, 32, 1, 1).shape)
# torch.Size([4, 1024, 2, 1])
print(b.repeat(4, 32, 2, 1).shape)
# torch.Size([4, 32, 1, 1])
print(b.repeat(4, 1 , 1, 1).shape)
# torch.Size([4, 32, 32, 32])
print(b.repeat(4, 1 , 32, 32).shape)

print('---------------------------------------------------')
# torch.Size([4, 32, 14, 14])
print(b.expand(4, 32, 14, 14).shape)
# torch.Size([1, 32, 1, 1])
print(b.expand(-1, 32, -1, -1).shape)
# torch.Size([1, 32, 1, -4])
print(b.expand(-1, 32, -1, -4).shape)
# torch.Size([1, 32, 1, 4])
print(b.expand(-1, 32, -1, 4).shape)
