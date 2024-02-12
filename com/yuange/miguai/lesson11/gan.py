import torch
from torch import nn, optim, autograd
import numpy as np
import visdom
from torch.nn import functional as F
from matplotlib import pyplot as plt
import random

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

h_dim = 400
batchsz = 512
viz = visdom.Visdom()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True), # 当参数为 True 时，ReLU 函数将进行inplace操作，即直接修改输入而不分配新的内存空间。
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2)
        )

    def forward(self, z):
        output = self.net(z)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        # 将 output 展平为1维张量并返回
        return output.view(-1)

def data_generator():
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    while True:
        dataset = []
        for i in range(batchsz):
            point = np.random.randn(2) * .02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414
        """
        生成器函数可以通过 yield 语句多次产生值，每次产生一个值后函数会被挂起，并且保留其状态，直到下一次请求一个新的值。
        """
        yield dataset

def generate_image(D, G, xr, epoch):
    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    # 创建一个等差数列
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    # 重新塑造（reshape）数组的形状
    points = points.reshape((-1,2))

    with torch.no_grad():
        points = torch.Tensor(points).cuda()
        disc_map = D(points).cpu().numpy()
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)

    with torch.no_grad():
        z = torch.randn(batchsz, 2).cuda()
        samples = G(z).cpu().numpy()
    xr = xr.cpu().numpy()
    plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    viz.matplot(plt, win='contour', opts=dict(title='p(x):%d'%epoch))

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

def main():
    torch.manual_seed(23)
    np.random.seed(23)

    G = Generator().cuda()
    D = Discriminator().cuda()
    # 初始化模型权重
    G.apply(weights_init)
    D.apply(weights_init)

    # betas=(0.5, 0.9)：用于计算梯度的一阶矩估计和二阶矩估计的系数
    optim_G = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))

    data_iter = data_generator()
    # (512, 2)
    print('batch:', next(data_iter).shape)

    viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G']))

    for epoch in range(50000):
        for _ in range(5):
            x = next(data_iter)
            xr = torch.from_numpy(x).cuda()

            predr = D(xr)
            lossr = - (predr.mean())

            z = torch.randn(batchsz, 2).cuda()
            xf = G(z).detach()
            predf = D(xf)
            lossf = predf.mean()

            loss_D = lossr + lossf
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        z = torch.randn(batchsz, 2).cuda()
        xf = G(z)
        predf = D(xf)
        loss_G = - (predf.mean())
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
            generate_image(D, G, xr, epoch)
            print(loss_D.item(), loss_G.item())

if __name__ == '__main__':
    main()