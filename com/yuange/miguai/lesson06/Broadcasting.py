import torch

# 创建形状为 (3, 4) 的张量 a 和一个标量 b
a = torch.randn(3, 4)
b = torch.tensor(3.)

# 对张量 a 和标量 b 进行逐元素相乘，使用广播机制
c = a * b

print(a, '\n')
print(b, '\n')
print(c, '\n')
