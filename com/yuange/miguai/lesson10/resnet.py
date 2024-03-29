import torch
from torch import nn
from torch.nn import functional as F

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 残差连接
        out = self.extra(x) + out
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_class):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16)
        )
        self.blk1 = ResBlk(16, 32, stride=3)
        self.blk2 = ResBlk(32, 64, stride=3)
        self.blk3 = ResBlk(64, 128, stride=2)
        self.blk4 = ResBlk(128, 256, stride=2)
        self.outlayer = nn.Linear(256 * 3 * 3, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x

def main():
    blk = ResBlk(64, 128)
    tmp = torch.randn(2, 64, 224, 224)
    out = blk(tmp)
    # torch.Size([2, 128, 224, 224])
    print('block:', out.shape)

    model = ResNet18(5)
    tmp = torch.randn(2, 3, 224, 224)
    out = model(tmp)
    # torch.Size([2, 5])
    print('resnet:', out.shape)

    p = sum(map(lambda p:p.numel(), model.parameters()))
    # 1233925
    print('parameters size:', p)

if __name__ == '__main__':
    main()