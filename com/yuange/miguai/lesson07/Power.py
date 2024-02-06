import torch

"""
full() 是 PyTorch 中张量（Tensor）对象的一个方法，用于创建一个指定形状和数值的张量。
full() 方法接受三个参数：size、fill_value 和 dtype。size 表示要创建的张量的形状，fill_value 表示要填充到张量中的数值，dtype 表示要创建的张量的数据类型，默认为 None，表示使用默认数据类型。
"""
a = torch.full([2,2], 3)
# tensor([[3, 3],
#         [3, 3]])
print(a)
"""
a.pow() 是 PyTorch 中用于计算张量元素的指数幂的函数。它可以对张量中的每个元素进行指数幂运算，并返回一个具有相同形状的新张量。
"""
# tensor([[9, 9],
#         [9, 9]])
print(a.pow(2))
# tensor([[9, 9],
#         [9, 9]])
print(a**2)
aa = a**2
"""
torch.sqrt() 是 PyTorch 中用于计算张量元素的平方根的函数。它可以对张量中的每个元素进行平方根运算，并返回一个具有相同形状的新张量。
"""
# tensor([[3., 3.],
#         [3., 3.]])
print(aa.sqrt())
"""
torch.rsqrt() 是 PyTorch 中用于计算张量元素的倒数平方根（reciprocal square root）的函数。它可以对张量中的每个元素进行倒数平方根运算，并返回一个具有相同形状的新张量。
"""
# tensor([[0.3333, 0.3333],
#         [0.3333, 0.3333]])
print(aa.rsqrt())
# tensor([[3., 3.],
#         [3., 3.]])
print(aa ** (0.5))