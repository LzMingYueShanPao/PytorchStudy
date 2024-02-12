import matplotlib.pyplot as plt
import torch
from torch import nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        x.shape[1:] 表示张量 x 的形状中从第二个维度开始（即索引为 1 的维度）到最后一个维度的切片。这样做的目的是获取除第一个维度外的所有其他维度的大小。
        torch.prod() 函数用于计算给定张量中所有元素的乘积。在这里，它被用于计算 x.shape[1:] 中所有维度大小的乘积。
        item() 方法用于获取张量中的单个值，并将其转换为 Python 中的标量值。
        """
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xtitle([])
        plt.ytitle([])
    plt.show()