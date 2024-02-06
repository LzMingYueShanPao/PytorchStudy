import torch

src = torch.tensor([[4,3,5],
                    [6,7,8]])
# 获取索引为0 2 5 的数据
# tensor([4, 5, 8])
print(torch.take(src, torch.tensor([0, 2, 5])))

print('----------------------------------------------------------')
x = torch.randn(3,4)
# tensor([[-0.4389, -0.1662,  0.2082,  0.7776],
#         [ 0.1808, -0.2395, -0.1150,  1.0289],
#         [-0.5516, -1.5390,  0.2826, -0.2452]])
print(x)
mask = x.ge(0.5)
# tensor([[False, False, False,  True],
#         [False, False, False,  True],
#         [False, False, False, False]])
print(mask)
# tensor([0.7776, 1.0289])
print(torch.masked_select(x, mask))
# torch.Size([2])
print(torch.masked_select(x, mask).shape)

print('----------------------------------------------------------')
a = torch.rand(4, 3, 28, 28)
# torch.Size([4, 3, 28, 28])
print(a.shape)
# torch.Size([4, 3, 28, 28])
print(a[...].shape)
# 选取第一个维度的 0索引数据，以及下面维度的全部数据
# torch.Size([3, 28, 28])
print(a[0,...].shape)
# 选取第一个维度上的全部数据下的第二个维度的 1索引数据，以及下面维度的全部数据
# torch.Size([4, 28, 28])
print(a[:,1,...].shape)
# 选取第一个维度上的全部数据下的第二个维度的 [0,2] 索引数据，不包括2，以及下面维度的全部数据
# torch.Size([4, 2, 28, 28])
print(a[:,:2].shape)

print('----------------------------------------------------------')
# 从第一个维度中取出 [0,2] 索引，不包括2 的数据
# torch.Size([2, 3, 28, 28])
print(a.index_select(0,torch.tensor([0,2])).shape)
# 从第二个维度中获取 [1,2] 索引，包括2 的数据
# torch.Size([4, 2, 28, 28])
print(a.index_select(1,torch.tensor([1,2])).shape)
# 从第二个维度中获取 [0,1] 索引，包括1 的数据
# torch.Size([4, 2, 28, 28])
print(a.index_select(1,torch.tensor([0,1])).shape)
# 从第三个维度中随机取28个数据
# torch.Size([4, 3, 28, 28])
print(a.index_select(2,torch.arange(28)).shape)
# 从第三个维度中随机取8个数据
# torch.Size([4, 3, 8, 28])
print(a.index_select(2,torch.arange(8)).shape)
# 从第四个维度中随机取9个数据
# torch.Size([4, 3, 28, 9])
print(a.index_select(3,torch.arange(9)).shape)


print('----------------------------------------------------------')
# torch.Size([4, 3, 14, 14])
print(a[:,:,0:28:2,0:28:2].shape)
# torch.Size([4, 3, 14, 14])
print(a[:,:,::2,::2].shape)

print('----------------------------------------------------------')
# torch.Size([4, 3, 28, 28])
print(a.shape)
# torch.Size([2, 3, 28, 28])
print(a[:2].shape)
# torch.Size([2, 1, 28, 28])
print(a[:2,:1,:,:].shape)
# torch.Size([2, 2, 28, 28])
print(a[:2,1:,:,:].shape)
# torch.Size([2, 1, 28, 28])
print(a[:2,-1:,:,:].shape)
# torch.Size([2, 3, 28, 28])
print(a[:2,-3:,:,:].shape)

print('----------------------------------------------------------')
# 访问张量 a 的第一个元素（即第 0 个元素）
# torch.Size([3, 28, 28])
print(a[0].shape)
# 访问张量 a 的第一个元素（即第 0 个元素）下的第一个元素
# torch.Size([28, 28])
print(a[0,0].shape)
# 访问张量 a 的第一个元素（即第 0 个元素）下的第一个元素下的第3个元素下的第四个元素
# tensor(0.4999)
print(a[0,0,2,4])