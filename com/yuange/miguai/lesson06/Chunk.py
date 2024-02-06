import torch

a = torch.rand(32, 8)
b = torch.rand(32, 8)
c = torch.stack([a,b], dim=0)
# torch.Size([2, 32, 8])
print(c.shape)
"""
chunk() 是 PyTorch 中张量（Tensor）对象的一个方法，用于将张量沿着指定的维度进行分块（Chunk）。它将张量按照给定的块数进行平均分割，并返回一个包含分块结果的列表。
chunk() 方法接受两个参数：chunks 和 dim。chunks 表示要分成的块数，dim 表示要沿着哪个维度进行分块。
需要注意的是，使用 chunk() 方法时，被分块的维度长度必须能够被块数整除，否则会抛出错误。另外，分块操作并不会复制张量的数据，而是返回原始张量的视图（View），即共享内存。
"""
# ValueError: not enough values to unpack (expected 2, got 1)
# aa, bb = c.split(2, dim=0)
aa, bb = c.chunk(2, dim=0)
# torch.Size([1, 32, 8]) torch.Size([1, 32, 8])
print(aa.shape, bb.shape)
aa, bb = c.chunk(2, dim=1)
# torch.Size([2, 16, 8]) torch.Size([2, 16, 8])
print(aa.shape, bb.shape)