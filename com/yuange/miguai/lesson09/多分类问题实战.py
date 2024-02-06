import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

batch_size = 200
learning_rate = 0.01
epochs = 10

device = torch.device('cuda:0')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True
)

w1, b1 = torch.randn(200, 784, requires_grad=True), torch.zeros(200, requires_grad=True)
w2, b2 = torch.randn(200, 200, requires_grad=True), torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(10, 200, requires_grad=True), torch.zeros(10, requires_grad=True)

"""
Kaiming 初始化方法也被称为 He 初始化，它是一种适用于使用 ReLU 激活函数的神经网络的权重初始化方法。
该方法根据输入和输出的维度自适应地调整权重的初始化方式，有助于缓解梯度消失或梯度爆炸问题，并提高网络的训练效果。
"""
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)

def forward(x):
    x = x@w1.t() + b1
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x@w3.t() + b3
    x = F.relu(x)
    return x

"""
SGD 每次迭代从训练集中随机选择一个样本，计算其梯度，并使用梯度对模型参数进行更新。
这样做的好处是减少了每次迭代的计算开销，并且可以更快地更新模型参数。
"""
optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 第一个维度的大小由 -1 自动确定，即数据本身的维度
        data = data.view(-1, 28*28)
        logist = forward(data)
        loss = criteon(logist, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('训练 Epoch：{} [{}/{} ({:.0f}%)]\t损失：{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()
            ))
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28*28)
        logist = forward(data)
        test_loss += criteon(logist, target).item()

        # max() 会返回输入张量中每行的最大值以及对应的索引。max(1)[1]表示只返回最大值的索引
        pred = logist.data.max(1)[1]
        correct += pred.eq(target.data).sum()
    test_loss /= len(test_loader.dataset)
    print('\n测试数据集：平均损失：{:.4f}，准确率：{}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    ))