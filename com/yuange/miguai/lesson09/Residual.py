import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        """
        构建神经网络模型的容器，可以将多个层组合成一个序列。这个容器能够按照传入的顺序依次执行每个层，从而构建一个简单的、线性的神经网络模型。
        可以将各种层（比如全连接层、卷积层、池化层等）按顺序添加到容器中，创建一个包含这些层的模型。
        虽然 nn.Sequential() 本身并不是专门用来实现残差连接的，但通过结合自定义的残差块类和 nn.Sequential()，
        我们可以很方便地构建包含残差连接的复杂神经网络模型。
        """
        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out