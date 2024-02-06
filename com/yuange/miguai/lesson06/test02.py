import torch
import numpy as np

# torch.randn(3, 3) 是 PyTorch 中的一个函数，主要用于创建指定形状的张量，并且张量中的元素是从标准正态分布（均值为 0，方差为 1）中抽取的随机数。
# tensor([[ 0.1841,  0.6816,  0.0501],
#         [ 0.9872, -0.5350,  0.5579],
#         [ 0.1218,  0.2478,  0.7946]])
print(torch.randn(3, 3))
"""
torch.normal() 是 PyTorch 中的一个函数，用于从给定的均值和标准差参数中抽取随机数，生成符合正态分布（高斯分布）的张量。

下面例子创建了一个形状为 (10,) 的张量，其中的每个元素都是从指定的正态分布中抽取的随机数。具体来说，该函数使用了以下参数：
    mean：均值参数，通过 torch.full([10],0) 创建了一个形状为 (10,) 的张量，其中的所有元素都是 0。
    std：标准差参数，通过 torch.arange(1, 0, -0.1) 创建了一个形状为 (10,) 的张量，其中的元素依次从 1 开始递减到 0，间隔为 0.1。
    
torch.normal() 函数生成的随机数是独立地抽取的，并且每个元素的值都是从对应位置的正态分布中抽取得到的。
"""
print(torch.normal(mean=torch.full([10],0), std=torch.arange(1, 0, -0.1)))

print(torch.normal(mean=torch.full([10],0), std=torch.arange(1, 0, -0.1)))


print('----------------------------------------------------------')
# torch.rand(3, 3) 是 PyTorch 中的一个函数，主要用于创建指定形状的张量，并且张量中的元素是在区间 [0, 1) 内均匀分布的随机数。
# tensor([[0.6198, 0.5361, 0.3138],
#         [0.9083, 0.4252, 0.6370],
#         [0.9790, 0.8013, 0.6170]])
print(torch.rand(3, 3))
a = torch.rand(3, 3)
# torch.rand_like(a) 会返回一个新的张量，其形状与张量 a 相同，并且其中的元素值是在区间 [0, 1) 内均匀分布的随机数。
print(torch.rand_like(a))
# torch.randint(1, 10, [3, 3]) 会返回一个新的大小为 3x3 的整数张量，其中的元素值是从1到9之间的随机整数。
# tensor([[1, 3, 2],
#         [6, 6, 9],
#         [4, 8, 2]])
print(torch.randint(1, 10, [3, 3]))


print('----------------------------------------------------------')
print(torch.tensor([1.2, 3]).type())
"""
自PyTorch 2.1起，torch.set_default_tensor_type（）已弃用，请使用torch.set_default_dtype（）和torch.set-default_device（）作为替代方案。
"""
# 打印：torch.DoubleTensor
#torch.set_default_tensor_type(torch.DoubleTensor)
# 打印：torch.DoubleTensor
torch.set_default_dtype(torch.double)
print(torch.tensor([1.2, 3]).type())

print('----------------------------------------------------------')
# torch.empty() 是 PyTorch 中用于创建一个未初始化张量的方法。与 torch.zeros() 和 torch.ones() 等方法不同，torch.empty() 方法不会将张量元素初始化为 0 或 1。
# 具体来说，它会创建一个指定大小和数据类型的空张量，并将其元素值设置为内存中随机存在的任意值。
# tensor([0.])
print(torch.empty(1))
"""
这是一个构造函数，用于创建一个指定大小的张量。它会返回一个未初始化的张量，其元素值取决于内存中的随机数据。在使用时，可以根据需要使用其他方法或操作来填充或更改张量的元素值。
注意，torch.Tensor() 的参数是张量的大小，而不是张量的元素值。
"""
# tensor([[1.0400e+03, 1.9828e-42, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00]])
print(torch.Tensor(2, 3))
# tensor([[1149370656,       1415,          0],
#         [         0,          0,          0]], dtype=torch.int32)
print(torch.IntTensor(2, 3))
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])
print(torch.FloatTensor(2, 3))

print('----------------------------------------------------------')
"""
torch.tensor([2., 3.2]) 和 torch.FloatTensor([2., 3.2]) 都是将数据 [2., 3.2] 转换为 torch.Tensor 对象的方式，但两者有一些细微的区别。
torch.tensor([2., 3.2]) 是使用 torch.tensor() 函数创建 torch.Tensor 对象的一种方式。该函数会根据输入数据自动推断数据类型，并返回一个新的 torch.Tensor 对象。如果输入数据本身就是 torch.Tensor 对象，那么该函数会直接返回原始对象。
torch.FloatTensor([2., 3.2]) 是使用 torch.FloatTensor() 构造函数创建 torch.Tensor 对象的一种方式。该函数会将输入数据转换为浮点类型，并返回一个新的 torch.Tensor 对象。如果输入数据本身已经是 torch.Tensor 对象，那么该函数会返回一个新的 torch.Tensor 对象，而不是直接返回原始对象。
如果你不确定输入数据的类型，并且希望让 PyTorch 自动推断数据类型，那么使用 torch.tensor() 函数可能更好。如果你需要将输入数据显式地转换为浮点类型的 torch.Tensor 对象，那么使用 torch.FloatTensor() 构造函数可能更好。
"""
# tensor([2.0000, 3.2000])
print(torch.tensor([2., 3.2]))
# tensor([2.0000, 3.2000])
print(torch.FloatTensor([2., 3.2]))

# tensor([[ 2.0000,  3.2000],
#         [ 1.0000, 22.3000]])
print(torch.tensor([[2., 3.2], [1., 22.3]]))

print('----------------------------------------------------------')
# 创建了一个包含两个元素的 numpy 数组 a，其中第一个元素为整数 2，第二个元素为浮点数 3.3
a = np.array([2, 3.3])
# 使用 torch.from_numpy() 函数将 a 转换为 torch.Tensor 对象，并将结果打印。需要注意的是，torch.from_numpy() 函数会保留原始 numpy 数组的数据类型，并将其转换为对应的 torch.Tensor 数据类型。
# tensor([2.0000, 3.3000], dtype=torch.float64)
print(torch.from_numpy(a))

#使用 numpy 库创建了一个形状为 (2, 3) 的全1数组。
a = np.ones([2, 3])
# tensor([[1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64)
print(torch.from_numpy(a))