import torch.nn as nn
import torch

class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.net = nn.Linear(4, 3)

    def forward(self, x):
        return self.net(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(BasicNet(), nn.ReLU(), nn.Linear(3, 2))

    def forward(self, x):
        return self.net(x)

device = torch.device('cuda')
net = Net()
net.to(device)

# train
net.train()
# ...
# test
net.eval()



"""
从文件 'ckpt.mdl' 中加载保存的模型状态字典。通常，这个文件包含了模型的参数（例如权重和偏置），但不包括模型的架构定义。
将加载的状态字典应用到模型 net 上。这个操作会将所有的参数（权重和偏置等）按照名称和维度正确地加载到模型的对应层中。
为了这一步能够成功，net 必须已经被初始化为与保存状态时相同的模型架构。
"""
# net.load_state_dict(torch.load('ckpt.mdl'))

# train ...

"""
训练完成后，可以通过调用 torch.save 函数并传递 net.state_dict() 来保存模型的状态字典。
这会将模型的参数（权重和偏置等）保存到 'ckpt.mdl' 文件中。这样，你就可以在未来的时间点重新加载这些参数，无需重新训练模型。
"""
torch.save(net.state_dict(), 'ckpt.mdl')