import torch
import time

CUDA_LAUNCH_BLOCKING=1

print(torch.__version__)
print(torch.cuda.is_available())

# print('hello world')

"""
在 PyTorch 中，torch.randn() 是一个用于生成服从标准正态分布（均值为 0，方差为 1）的随机数的函数。它的作用是创建一个指定大小的张量（tensor），其中的元素值是从标准正态分布中随机抽取的。

torch.randn() 函数的语法如下：

python
torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
参数说明：

*size：用于指定生成的随机张量的大小。可以是以逗号分隔的整数序列，也可以是一个可迭代对象。
out：可选参数，用于存储结果的输出张量。
dtype：可选参数，用于指定输出张量的数据类型。
layout：可选参数，用于指定输出张量的布局。
device：可选参数，用于指定输出张量所在的设备。
requires_grad：可选参数，用于指定输出张量是否需要梯度计算。
"""
a = torch.randn(10000,1000)
b = torch.randn(1000,2000)

#计算耗时情况
t0 = time.time()
"""
torch.matmul(a, b) 是 PyTorch 中的矩阵乘法函数，用于计算两个张量的矩阵乘积。

该函数的语法如下：

python
torch.matmul(a, b, out=None)
参数说明：

a：要进行矩阵乘法的第一个输入张量。
b：要进行矩阵乘法的第二个输入张量。
out：可选参数，用于存储结果的输出张量。
注意事项：

a 和 b 的形状必须满足矩阵乘法的规则。具体而言，a 的最后一个维度的大小必须等于 b 的倒数第二个维度的大小。
如果 a 和 b 是一维张量，则执行向量的点积运算。
"""
c = torch.matmul(a,b)
t1 = time.time()
"""
在PyTorch中，norm函数用于计算张量的范数（norm）值。范数是一个度量向量大小的指标，它衡量了向量的长度或大小。
norm函数有以下作用：
计算L1范数：torch.norm(input, p=1)
L1范数是指将向量中每个元素的绝对值相加，得到的结果。在机器学习中，L1范数常用于稀疏性推断和特征选择等任务。
计算L2范数：torch.norm(input, p=2)
L2范数是指将向量中每个元素的平方和开根号，得到的结果。在机器学习中，L2范数常用于正则化项、损失函数等地方。它可以惩罚较大的权重值，避免模型过拟合。
计算无穷范数：torch.norm(input, p=float('inf'))
无穷范数是指向量中所有元素绝对值的最大值。它表示向量中最大的绝对值元素的大小。
计算其他范数：torch.norm(input, p=...)
p参数可以是任意正数，用于计算自定义的范数。例如，当p=0.5时，计算的是L0.5范数，用于强调向量中较大的元素。
norm函数还可以用于计算矩阵的范数，通过指定dim参数进行维度上的计算。例如，torch.norm(input, p=2, dim=1)可以计算矩阵每行的L2范数。
需要注意的是，norm函数返回的是一个标量值，而不是保留输入张量的形状。如果需要计算张量的范数，并保持其形状，可以使用torch.norm(input, p=2, dim=..., keepdim=True)。
"""
print(a.device,t1 - t0,c.norm(2))

"""
在PyTorch中,设备(device)用来指定计算操作在哪个硬件上执行,例如CPU或GPU
"""
# device = torch.device('cuda')
device = torch.device('cpu')
"""
张量.to(device) 是一个用于将张量(tensor)移动到指定设备(device)上的方法。它可以将张量从一种设备（如 CPU）转移到另一种设备（如 GPU），或者在不同的 GPU 设备之间进行转移。

该方法的语法如下：

python
tensor.to(device=None, dtype=None, non_blocking=False)
参数说明：

device：指定要移动到的目标设备。可以是字符串（如 "cpu" 或 "cuda:0"）或 torch.device 对象。
dtype：可选参数，用于指定转移后的张量的数据类型。
non_blocking：可选参数，如果设置为 True，则表示异步转移。默认为 False。
"""
a = a.to(device)
b = b.to(device)

t0 = time.time()
c = torch.matmul(a,b)
t2 = time.time()
print(a.device,t2-t0,c.norm(2))

t0 = time.time()
c = torch.matmul(a,b)
t2 = time.time()
print(a.device,t2-t0,c.norm(2))



