import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

batch_size = 200
learning_rate = 0.01
epochs = 10

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.Dropout(0.5), #有50%的神经元下降
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.Dropout(0.5),  # 有50%的神经元下降
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x

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

device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.78 ,weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, 'min')
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.to(device)

        regularization_loss = 0
        for param in net.parameters():
            regularization_loss += torch.sum(torch.abs(param))

        logist = net(data)
        loss = criteon(logist, target)

        loss = loss + 0.01 * regularization_loss

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
    data, target = data.to(device), target.cuda()
    logist = net(data)
    test_loss += criteon(logist, target).item()

    pred = logist.argmax(dim=1)
    correct += pred.eq(target).float().sum().item()

test_loss /= len(test_loader.dataset)
print('\n测试数据集：平均损失：{:.4f}，准确率：{}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
))