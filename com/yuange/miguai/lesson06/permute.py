import torch

a = torch.rand(4, 3, 28, 28)
# torch.Size([4, 28, 28, 3])
print(a.transpose(1,3).shape)
torch.tensor()

b = torch.rand(4, 3, 28, 32)
# torch.Size([4, 32, 28, 3])
print(b.transpose(1,3).shape)
# torch.Size([4, 28, 32, 3])
print(b.transpose(1,3).transpose(1,2).shape)
"""
permute() 是 PyTorch 中张量（Tensor）对象的一个方法，用于对张量进行维度重排（Permutation），即按照指定的顺序重新排列张量的维度。
permute() 方法接受一个整数列表作为参数，列表中的每个整数表示要将对应位置的维度移动到的新位置。通过这种方式，可以灵活地重排张量的维度顺序。
"""
# torch.Size([4, 28, 32, 3])
print(b.permute(0,2,3,1).shape)
# RuntimeError: permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e.input.dim() = 4 is not equal to len(dims) = 3
# print(b.permute(0,2,1).shape)