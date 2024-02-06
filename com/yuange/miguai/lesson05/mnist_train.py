import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt

from utils import plot_image, plot_curve, one_hot

batch_size = 512

##1.加载数据
"""
torch.utils.data.DataLoader是PyTorch库中的一个类，用于加载和批量处理数据。它提供了对数据集的高效迭代方式，并支持多线程数据加载、数据打乱和批量处理等功能。
在深度学习任务中，我们通常需要将大量的训练数据划分为小批量进行训练，以提高模型的训练效率和泛化能力。DataLoader类可以帮助我们实现这个目标，它可以接受一个数据集对象（如torch.utils.data.Dataset的子类）作为输入，并提供方便的数据加载和处理接口。
使用DataLoader类的一般步骤如下：
    创建数据集对象：首先，我们需要创建一个数据集对象，该对象包含要加载和处理的数据。
    创建DataLoader对象：然后，我们可以使用数据集对象来创建一个DataLoader对象，该对象负责加载和处理数据。
    迭代数据：最后，我们可以使用DataLoader对象来迭代加载和处理数据集。每次迭代会返回一个批量的数据。
"""
"""
torchvision.datasets.MNIST 是 PyTorch 框架中的一个图像数据集，包含了手写数字图片和对应的标签。这个数据集是从 NIST 的特定任务中提取出来的，其中的手写数字图片被广泛用于机器学习领域的测试和验证。
torchvision.datasets.MNIST 的语法如下：
    torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
参数说明：
    root：数据集的根目录，在该目录下会自动下载 MNIST 数据集。
    train：True 表示返回训练集数据，False 表示返回测试集数据，默认为 True。
    transform：对数据集进行预处理的函数或变换。
    target_transform：对标签进行预处理的函数或变换。
    download：是否下载数据集，默认为 False。
"""
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data2', train=True, download=True,
                                                                      transform=torchvision.transforms.Compose([
                                                                          torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                                                      ])),
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data2/', train=False, download=True,
                                                                      transform=torchvision.transforms.Compose([
                                                                          torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                                                      ])),
                                           batch_size=batch_size,
                                           shuffle=False)

"""
next() 是 Python 中的一个内置函数，用于获取迭代器的下一个元素。
    迭代器是一个实现了 __iter__() 和 __next__() 方法的对象。__iter__() 方法返回迭代器对象本身，而 __next__() 方法返回迭代器中的下一个元素。通过调用 next() 函数，可以不断获取迭代器中的下一个元素，直到迭代器结束或者触发 StopIteration 异常。
示例用法：
    my_list = [1, 2, 3, 4, 5]
    my_iter = iter(my_list)  # 获取迭代器
    
    print(next(my_iter))  # 输出 1
    print(next(my_iter))  # 输出 2
    print(next(my_iter))  # 输出 3
"""
x, y = next(iter(train_loader))
"""
x.shape 是一个 Numpy 数组或张量（如 PyTorch 或 TensorFlow）的属性，用于获取数组或张量的形状。
形状描述了数组或张量的维度信息，即它们在每个维度上的大小。通常，形状是一个元组（tuple），其中每个元素表示对应维度的大小。
x.min() 是一个 Numpy 数组或张量（如 PyTorch 或 TensorFlow）的方法，用于获取数组或张量中的最小值。
"""
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')

"""
nn.Module 是 PyTorch 中的一个基类，用于构建神经网络模型。在深度学习中，我们常常使用 nn.Module 类来定义自己的网络模型。
nn.Module 提供了一些常用的方法和属性，方便我们构建和管理模型。下面是 nn.Module 类的一些常用方法和属性：
    __init__(): 初始化方法，用于定义网络结构和初始化参数。
    forward(): 前向传播方法，定义模型的前向计算逻辑。
    parameters(): 返回模型中所有可学习参数的迭代器。
    named_parameters(): 返回模型中所有可学习参数及其名称的迭代器。
    state_dict(): 返回模型的状态字典，包含模型的所有参数。
    load_state_dict(): 加载模型的状态字典，用于恢复模型的参数。
    to(): 将模型移动到指定的设备（如 CPU 或 GPU）。
    train(): 将模型设置为训练模式，启用 Batch Normalization 和 Dropout 等训练相关的操作。
    eval(): 将模型设置为评估模式，关闭 Batch Normalization 和 Dropout 等训练相关的操作。
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        """
        nn.Linear 是 PyTorch 中的一个类，用于创建线性（全连接）层。
        线性层是神经网络中常见的一种层类型，它将输入数据与权重进行线性变换，并可选择是否添加偏置。线性层可以看作是一个矩阵乘法操作，其中输入数据的每个特征都与对应的权重相乘，然后求和得到输出。
        nn.Linear 的初始化参数通常为输入特征的数量和输出特征的数量，即线性层的输入维度和输出维度。在创建 nn.Linear 对象时，系统会自动随机初始化权重和偏置。
        示例用法：
            import torch
            import torch.nn as nn
            # 创建一个线性层，输入维度为3，输出维度为2
            linear = nn.Linear(3, 2)
            # 随机生成输入数据
            input_data = torch.randn(4, 3)  # 输入数据维度为 (4, 3)
            # 进行线性变换
            output = linear(input_data)  # 输出维度为 (4, 2)
        """
        # xw+b
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1 + b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2 + b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3 + b3
        x = self.fc3(x)

        return x

net = Net()
# [w1, b1, w2, b2, w3, b3]
# print("net.parameters()=",net.parameters())
# for param in net.parameters():
#     print(type(param), param.size(), param)
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

train_loss = []
for epoch in range(3):
    """
    enumerate() 是 Python 内置函数之一，用于枚举一个可迭代对象中的元素，并返回每个元素对应的索引值和值本身。通常情况下，我们会将 enumerate() 与 for 循环一起使用，以便按顺序访问可迭代对象中的每个元素。
    示例用法：
        fruits = ['apple', 'banana', 'orange']
        for index, value in enumerate(fruits):
            print(index, value)
    """
    for batch_size, (x,y) in enumerate(train_loader):
        # x: [b, 1, 28, 28], y: [512]
        # [b, 1, 28, 28] => [b, 784]
        """
        x.view()是PyTorch张量（tensor）的一个方法，用于改变张量的形状而不改变数据本身。它可以用于调整张量的维度、大小和顺序。
        在深度学习任务中，我们经常需要对张量进行形状的变换，以适应不同的模型或算法需求。view()方法提供了一种方便的方式来实现这个目标。
        view()方法的使用非常简单，只需要传入一个表示目标形状的元组即可。下面是一个示例：
            import torch
            # 创建一个3x4的张量
            x = torch.tensor([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]])
            # 改变张量的形状为2x6
            y = x.view(2, 6)
            print(y)
            输出:
            tensor([[ 1,  2,  3,  4,  5,  6],
                    [ 7,  8,  9, 10, 11, 12]])
        """
        x = x.view(x.size(0), 28*28)
        # => [b, 10]
        out = net(x)
        # [b, 10]
        y_onehot = one_hot(y)
        # loss = mse(out, y_onehot)
        """
        F.mse_loss 是 PyTorch 中的一个函数，用于计算均方误差损失（Mean Squared Error Loss）。均方误差是衡量预测值与目标值之间差异的常用指标，它计算预测值与目标值之间差的平方的平均值。
        函数签名：
            mse_loss(input, target, size_average=None, reduce=None, reduction='mean')
        参数说明：
            input: 预测值（模型的输出）。
            target: 目标值（真实标签）。
            size_average（已废弃）: 是否对每个样本的损失进行平均，默认为 None。
            reduce（已废弃）: 是否对所有样本的损失进行求和，默认为 None。
            reduction: 指定如何计算损失的缩减方式，可取值为 'mean'、'sum' 或 'none'，默认为 'mean'。如果设置为 'mean'，则返回所有样本均方误差的平均值；如果设置为 'sum'，则返回所有样本均方误差的总和；如果设置为 'none'，则返回每个样本的均方误差。
        """
        loss = F.mse_loss(out, y_onehot)
        """
        optimizer.zero_grad() 是 PyTorch 中用于将模型参数的梯度归零的方法。在训练神经网络时，我们通常会进行前向传播和反向传播计算梯度。在每次迭代中，我们需要清零上一次迭代中计算得到的参数梯度，以免梯度累积影响下一次迭代的结果。
        具体来说，optimizer.zero_grad() 方法会将优化器对象中所有参数的梯度置零。这样，在下一次进行反向传播前，就可以确保新的参数梯度不受之前的迭代影响。
        """
        optimizer.zero_grad()
        """
        自动计算当前损失函数对所有需要梯度求导的模型参数的梯度
        """
        loss.backward()
        # w' = w - (lr * grad)
        """
        调用优化器的 step() 方法，根据梯度更新模型参数。
        """
        optimizer.step()

        train_loss.append(loss.item())

        if batch_size % 10 == 0:
            print(epoch, batch_size, loss.item())

plot_curve(train_loss)
# 得到了比较好的参数 [w1 b1 w2 b2 w3 b3]

total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    # out: [b, 10] =>pred: [b]
    """
    out.argmax(dim=1) 是一个 PyTorch 中的操作，用于在指定维度上获取张量中最大值所在的索引。
    具体来说，假设 out 是一个张量（比如一个输出层的输出），通过调用 argmax(dim=1)，我们可以获取 out 张量每行中最大值所在的列索引，返回的结果是一个张量。
    下面是一个简单的示例：
        import torch
        out = torch.tensor([[0.2, 0.3, 0.5],
                            [0.6, 0.1, 0.3],
                            [0.4, 0.7, 0.5]])
        max_indices = out.argmax(dim=1)
        print(max_indices)
    输出结果为：
        tensor([2, 0, 1])
    """
    pred = out.argmax(dim=1)
    """
    eq() 是 PyTorch 中的一个张量操作，用于比较两个张量的元素是否相等。它返回一个新的布尔类型（bool）的张量，其中每个元素表示对应位置上的两个输入张量元素是否相等。
    以下是 eq() 的使用示例：
        import torch
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([1, 4, 3])
        result = x.eq(y)
        print(result)
    输出结果为：
        tensor([ True, False,  True])
    """
    """
    sum() 是 PyTorch 中的一个张量操作，用于计算张量中所有元素的总和。
    以下是 sum() 的使用示例：
        import torch
        x = torch.tensor([1, 2, 3, 4, 5])
        result = x.sum()
        print(result)
    输出结果为：
        tensor(15)
    """
    """
    float() 是 Python 内置的函数，用于将一个对象转换为浮点数类型。
    """
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc: ', acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28*28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')

