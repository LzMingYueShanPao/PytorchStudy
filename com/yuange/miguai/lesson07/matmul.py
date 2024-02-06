import torch

a = torch.full((2,2), 3.)
# tensor([[3., 3.],
#         [3., 3.]])
print(a)
b = torch.ones(2,2)
# tensor([[1., 1.],
#         [1., 1.]])
print(b)
"""
torch.mm(a, b) 是 PyTorch 中的一个函数，用于计算两个矩阵的乘积。其中，参数 a 和 b 分别为两个输入矩阵。
"""
# tensor([[6., 6.],
#         [6., 6.]])
print(torch.mm(a, b))
"""
首先，torch.mm() 函数只能用于计算两个二维矩阵的乘积，而且要求两个矩阵的形状必须满足乘积规则，即第一个矩阵的列数必须等于第二个矩阵的行数。例如，如果 a 的形状是 
(n,m)，b 的形状是(m,p)，那么就可以使用 torch.mm(a, b) 计算它们的乘积，结果的形状是(n,p)。
相比之下，torch.matmul() 函数更加灵活，可以用于计算不同维度的张量的乘积。具体来说，torch.matmul() 函数按照以下规则计算张量的乘积：
    如果两个输入都是一维张量，那么就进行点积运算。
    如果两个输入都是二维张量，那么就按照矩阵乘法规则计算它们的乘积。
    如果其中一个输入是一维张量，另一个输入是多维张量，那么就对一维张量进行广播，然后按照矩阵乘法规则计算它们的乘积。
    如果两个输入都是多维张量，那么就按照 Einstein 求和约定计算它们的乘积。这种方式可以处理不同维度的张量之间的乘积，例如批量矩阵乘法等。
除了输入形状和计算规则的不同之外，torch.mm() 和 torch.matmul() 在使用时还有一些细微的差别。具体来说，torch.mm() 只能接受两个参数，而且这两个参数必须都是 Tensor 类型；
而 torch.matmul() 可以接受多个参数，其中至少有一个是 Tensor 类型。此外，torch.matmul() 在计算时会自动进行类型转换和广播，而 torch.mm() 则需要保证输入的形状满足乘积规则，否则会抛出错误。
"""
# tensor([[6., 6.],
#         [6., 6.]])
print(torch.matmul(a, b))
"""
a @ b 语法是 torch.matmul() 函数的简写形式。它可以用于计算两个张量（包括标量、向量、矩阵等）的乘积，规则同 torch.matmul() 函数
"""
# tensor([[6., 6.],
#         [6., 6.]])
print(a@b)
a = torch.rand(4, 784)
x = torch.rand(4, 784)
w = torch.rand(512, 784)
# torch.Size([4, 512])
print((x@w.t()).shape)
a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
# RuntimeError: self must be a matrix（必须是矩阵）
# torch.mm(a,b).shape
# torch.Size([4, 3, 28, 32])
print(torch.matmul(a, b).shape)
b = torch.rand(4,1,64,32)
# torch.Size([4, 3, 28, 32])
print(torch.matmul(a, b).shape)
b= torch.rand(4,64,32)
# RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1（在非单例维度1上，张量a（3）的大小必须与张量b（4）的大小匹配）
# print(torch.matmul(a, b).shape)