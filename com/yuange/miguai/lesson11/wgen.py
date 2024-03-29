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

def gradient_penalty(D, xr, xf):
    LAMBDA = 0.3
    xf = xf.detach() # 将结果分离出计算图，以确保它们不参与梯度计算。
    xr = xr.detach()

    alpha = torch.rand(batchsz, 1).cuda()
    alpha = alpha.expand_as(xr) # 将当前张量扩展为与指定张量相同的形状。

    interpolates = alpha * xr + ((1 - alpha) * xf)
    interpolates.requires_grad_()

    disc_interpolates = D(interpolates)
    """
    outputs：指定需要对哪些张量求梯度，这里是 disc_interpolates。
    inputs：指定对哪些张量进行求导，这里是 interpolates。
    grad_outputs：指定输出张量的梯度，这里使用 torch.ones_like(disc_interpolates) 表示所有元素的梯度都为 1。
    create_graph：指定是否创建计算图，这里设置为 True，表示需要创建用于高阶求导的计算图。
    retain_graph：指定是否保留计算图，这里设置为 True，表示在多次调用 backward 时保留计算图。
    only_inputs：指定是否只对输入张量求导，这里设置为 True，表示只计算输入张量的梯度。
    """
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gp

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

            gp = gradient_penalty(D, xr, xf)

            loss_D = lossr + lossf + gp
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