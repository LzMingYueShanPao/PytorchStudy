import torch

a = torch.rand(32, 8)
b = torch.rand(32, 8)
c = torch.stack([a,b], dim=0)
# torch.Size([2, 32, 8])
print(c.shape)
# 将索引为0的[2]向量分解成2份
aa, bb = c.split([1,1], dim=0)
# torch.Size([1, 32, 8]) torch.Size([1, 32, 8])
print(aa.shape, bb.shape)
# 将索引为0的[2]向量分解成2份
aa, bb = c.split(1, dim=0)
# torch.Size([1, 32, 8]) torch.Size([1, 32, 8])
print(aa.shape, bb.shape)
# ValueError: not enough values to unpack (expected 2, got 1)
# aa, bb = c.split(2, dim=0)
d = torch.rand(6, 32, 8)
# 将索引为0的维度按2份切分，分成三份
aa, bb, cc = d.split(2, dim=0)
# torch.Size([2, 32, 8]) torch.Size([2, 32, 8]) torch.Size([2, 32, 8])
print(aa.shape, bb.shape, cc.shape)
# 将索引为0的维度按3份切分，分成2份
aa, bb = d.split(3, dim=0)
# torch.Size([3, 32, 8]) torch.Size([3, 32, 8])
print(aa.shape, bb.shape)