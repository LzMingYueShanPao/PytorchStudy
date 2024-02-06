import torch
import numpy as np

a = torch.randn(2,3,28,28)
# tensor([[[[-0.0396, -0.6517,  0.4871,  ..., -1.9813, -1.6730,  1.1297],
#           [-0.2390,  0.7178, -0.0231,  ..., -1.7410,  0.2839,  0.2333],
#           [-0.1811,  0.5589, -0.3689,  ...,  1.3239,  0.6439, -1.1037],
#           ...,
#           [-0.8081,  0.4995,  0.7416,  ...,  1.0837, -1.1337, -0.7126],
#           [ 0.5164,  1.5872,  0.2451,  ...,  1.0388, -0.1647,  0.2028],
#           [ 0.3426,  0.2086,  0.3570,  ..., -0.6927,  1.1706,  0.5738]],
#
#          [[-2.8317,  1.4574, -0.4800,  ..., -1.2562, -0.0448, -0.0371],
#           [-0.4545, -0.4266,  0.7207,  ..., -0.2422,  0.8742,  0.5116],
#           [ 0.0945, -0.5173,  0.8447,  ...,  0.4252, -0.2529,  0.9042],
#           ...,
#           [ 1.5553,  0.6381,  0.4920,  ...,  0.4263,  0.1135,  0.3261],
#           [ 0.6064,  1.4777,  0.9384,  ...,  2.7150,  0.2717,  1.3084],
#           [ 0.8381, -0.1196,  0.3270,  ..., -0.1663, -0.4318,  1.2343]],
#
#          [[-0.1660,  1.1196,  0.8238,  ...,  0.5664, -0.1033,  0.5293],
#           [-1.1351, -0.8023, -0.4372,  ...,  0.2246, -1.2505,  1.7081],
#           [-0.0991,  1.0825,  0.8596,  ..., -0.3852,  0.2745, -0.7484],
#           ...,
#           [-0.5133,  1.7538,  1.1862,  ..., -0.2098,  0.3438, -0.7807],
#           [-0.0481, -0.0631,  1.1632,  ..., -1.9394, -0.2255, -0.2427],
#           [-1.3879, -0.0079,  0.7696,  ..., -1.1836, -0.4040, -0.8613]]],
#
#
#         [[[-0.8392,  0.2016, -0.3291,  ..., -1.5058, -0.1530, -0.1398],
#           [ 0.0925, -1.1656,  0.7372,  ..., -0.5464,  0.0287, -0.4216],
#           [-1.0068, -1.8285,  0.1308,  ...,  0.3151,  0.3700, -0.2753],
#           ...,
#           [ 0.1038,  1.1298, -1.0560,  ...,  0.2318,  0.2229,  0.2806],
#           [-0.4278, -2.6615,  0.7045,  ..., -2.6037, -0.5293, -0.5962],
#           [ 0.6128,  2.1824, -1.9657,  ...,  0.2526, -1.5406, -1.1775]],
#
#          [[-2.0387, -0.0157, -0.5297,  ..., -1.1012,  0.6923,  0.7251],
#           [ 0.6034,  1.0818, -1.2802,  ...,  1.4034, -1.4950, -0.6461],
#           [ 1.1027, -0.1435, -0.7476,  ...,  1.0509, -0.3254, -0.9394],
#           ...,
#           [-0.1333,  0.1811,  0.0922,  ...,  3.3530, -0.1219,  0.2415],
#           [ 0.2790, -0.7276, -0.2038,  ...,  1.6598, -0.5150,  1.4421],
#           [ 0.7104,  1.0993, -0.3114,  ..., -0.3823, -0.3926,  1.1021]],
#
#          [[ 0.7400, -0.1806, -1.0639,  ...,  1.7429, -0.0608,  0.6155],
#           [ 0.6462, -0.5986, -0.4902,  ...,  1.1177,  1.0016, -0.2618],
#           [-0.3485, -0.0917, -0.5255,  ...,  0.1317,  0.1299,  0.2739],
#           ...,
#           [ 0.9073,  1.2008,  1.0540,  ..., -0.1261,  0.0160, -1.4382],
#           [ 0.3358,  0.1956, -0.9133,  ...,  0.9595,  1.9182,  0.9053],
#           [-0.5023, -0.0894, -0.7339,  ...,  1.0280, -1.6917, -1.2038]]]])
print(a)
# torch.Size([2, 3, 28, 28])
print(a.shape)
# a.numel() 是 PyTorch 中 torch.Tensor 对象的一个方法，用于计算张量中元素的总数。
# 28 * 28 * 3 * 2 = 4704
print(a.numel())
# a.dim() 是 PyTorch 中 torch.Tensor 对象的一个方法，用于计算张量的维度（也称为阶数或秩）。
# 4
print(a.dim())
a = torch.tensor(1)
# 0
print(a.dim())


print('----------------------------------------------------------')
a = torch.randn(1,2,3)
# tensor([[[ 0.6513,  1.2600,  0.0150],
#          [-0.4069,  1.0116, -0.9937]]])
print(a)
# torch.Size([1, 2, 3])
print(a.shape)
# tensor([[-0.7557, -0.3167,  0.9623],
#         [ 0.2060, -1.3954,  0.0914]])
print(a[0])
# [1, 2, 3]
print(list(a.shape))


print('----------------------------------------------------------')
a = torch.randn(2,3)
# tensor([[-0.3736,  0.4144,  0.1400],
#        [ 0.5257, -0.7232,  0.5419]])
print(a)
# torch.Size([2, 3])
print(a.shape)
# 2（两行）
print(a.size(0))
# 3（三列）
print(a.size(1))
# 2（两行）
print(a.shape[0])
# 3（三列）
print(a.shape[1])



print('----------------------------------------------------------')
# 创建一个包含两个元素且值全为 1 的一维数组
a = torch.ones(2)
# torch.Size([2])
print(a.shape)


print('----------------------------------------------------------')
# 将 [1.1] 作为参数传递给函数，它会自动推断出张量的数据类型，并创建一个包含单个浮点数值的一维张量。
# tensor([1.1000])
print(torch.tensor([1.1]))
# 将 [1.1, 2.2] 作为参数传递给函数，创建一个包含两个浮点数值 1.1 和 2.2 的一维张量。
# tensor([1.1000, 2.2000])
print(torch.tensor([1.1, 2.2]))
# 将数字 1 作为参数传递给函数，它会创建一个具有一个元素的张量，并将其初始化为零。
# tensor([0.])
print(torch.FloatTensor(1))
# 它会创建一个数据类型为 float32，大小为 2 的一维浮点型张量，并将其元素初始化为 0。
# tensor([0., 0.])
print(torch.FloatTensor(2))
# 创建一个包含两个元素且值全为 1 的一维数组。
# [1. 1.]
data = np.ones(2)
print(data)
# PyTorch 将 numpy 数组转换为 torch.Tensor 的函数
# tensor([1., 1.], dtype=torch.float64)
data = torch.from_numpy(data)
print(data)


print('----------------------------------------------------------')
# 创建一个标量浮点数张量 a
a = torch.tensor(2.2)
# a.shape 属性，它可以返回 PyTorch 张量的形状信息。形状是一个元组，其中包含张量在每个维度上的大小。例如，对于大小为 (3, 4) 的二维张量，其形状为 (3, 4)。
# 其形状为空元组 torch.Size([])
print(a.shape)
# 0
print(len(a.shape))
# torch.Size([])
print(a.size())


# torch.tensor() 是创建张量的函数之一。当传递一个 Python 标量值给 torch.tensor() 函数时，它会自动推断出张量的数据类型，并创建一个标量张量。
# 传入1.标量，得到一个tensor(1.)张量（Int类型）
print(torch.tensor(1.))
# 传入1.3标量，得到一个tensor(1.3000)张量（Float类型）
print(torch.tensor(1.3))


# 创建一个2行3列的Tensor
a = torch.randn(2,3)
# 查看a的类型：torch.FloatTensor
print(a.type())
# <class 'torch.Tensor'>
print(type(a))
# True
print(isinstance(a, torch.FloatTensor))