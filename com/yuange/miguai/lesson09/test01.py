import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
# print('hello world')

a = torch.full([4], 1/4.)
# tensor([0.2500, 0.2500, 0.2500, 0.2500])
print(a)

b = a*torch.log2(a)
# tensor([-0.5000, -0.5000, -0.5000, -0.5000])
print(b)
# tensor(2.)
print(-(b).sum())

a = torch.tensor([0.1, 0.1, 0.1, 0.7])
b = a*torch.log2(a)
# tensor(1.3568)
print(-(b).sum())

"""
概率值越大，经过log函数之后得到的熵值越小，分类越准确
"""
a = torch.tensor([0.001, 0.001, 0.001, 0.999])
b = a*torch.log2(a)
# tensor(0.0313)
print(-(b).sum())