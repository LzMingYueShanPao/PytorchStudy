import torch
from torch import autograd

"""
torch.tensor 是 PyTorch 中的一个函数，用于创建张量（Tensor）对象。张量是多维数组，类似于数学中的矩阵。

torch.tensor 的作用是根据给定的数据创建一个张量。它可以接受各种类型的数据作为输入，并自动推断出张量的数据类型和形状。

torch.tensor 的语法如下：

python
torch.tensor(data, dtype=None, device=None, requires_grad=False)
参数说明：

data：要转换为张量的数据。可以是列表、元组、NumPy 数组等。
dtype：可选参数，指定张量的数据类型。如果不指定，则会自动推断。
device：可选参数，指定张量存储的设备。默认为 CPU。
requires_grad：可选参数，指定是否需要计算梯度，默认为 False。
"""
x = torch.tensor(1.)
a = torch.tensor(1.,requires_grad=True)
b = torch.tensor(2.,requires_grad=True)
c = torch.tensor(3.,requires_grad=True)

y = a**2 * x + b * x + c

print('之前：',a.grad,b.grad,c.grad)
grads = autograd.grad(y,[a,b,c])
print('之后：',grads[0],grads[1],grads[2])
