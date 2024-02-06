import torch

a = torch.full([8], 1.)
b = a.view(2, 4)
c = a.view(2, 2, 2)
# tensor([1., 1., 1., 1., 1., 1., 1., 1.])
print(a)
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.]])
print(b)
# tensor([[[1., 1.],
#          [1., 1.]],
#
#         [[1., 1.],
#          [1., 1.]]])
print(c)
"""
norm() 是一个张量的方法，用于计算张量的范数（norm）。
norm() 方法的使用方式有两种：
    1.使用默认参数：如果不指定任何参数，norm() 方法将计算张量的二范数（Euclidean 范数）。
    2.指定参数：可以通过传递额外的参数来指定不同类型的范数。
        result = tensor.norm(p)
      其中，p 是一个整数或浮点数，用于指定要计算的范数类型。常见的范数类型包括：
        1: 一范数（Manhattan 范数）
        2: 二范数（Euclidean 范数）
        p: Lp 范数，其中 p 是一个正整数或浮点数
        inf: 无穷范数（向量中最大绝对值）
"""
# tensor(8.) tensor(8.) tensor(8.)
print(a.norm(1), b.norm(1), c.norm(1))
# tensor(2.8284) tensor(2.8284) tensor(2.8284)
print(a.norm(2), b.norm(2), c.norm(2))
# tensor([4., 4.])
print(b.norm(1, dim=1))
# tensor([2., 2.])
print(b.norm(2, dim=1))
# tensor([[2., 2.],
#         [2., 2.]])
print(c.norm(1, dim=0))
# tensor([[1.4142, 1.4142],
#         [1.4142, 1.4142]])
print(c.norm(2, dim=0))