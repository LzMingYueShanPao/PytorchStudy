import torch
from torch import nn
from torch.nn import functional as F

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        )

        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        tmp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        # torch.Size([2, 16, 5, 5])
        print('conv out.shape=',out.shape)

        # 使用交叉熵
        # self.criteon = nn.CrossEntropyLoss()

    def forward(self, x):
        batchsz = x.size(0)
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)
        # [b, 16, 5, 5] => [b, 16*5*5]
        # x = x.view(batchsz, -1)
        x = x.view(batchsz, 16 * 5 * 5)
        # [b, 16*5*5] => [b, 10]
        logits = self.fc_unit(x)
        # [b, 10]
        # pred = F.softmax(logits, dim=1)
        # loss = self.criteon(logits, y)
        return logits


def main():
    net = Lenet5()

    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    # torch.Size([2, 10])
    print('lenet out.shape=', out.shape)

if __name__ == '__main__':
    main()