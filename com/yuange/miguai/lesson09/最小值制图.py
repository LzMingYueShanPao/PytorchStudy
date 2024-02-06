import torch
import numpy as np
import matplotlib.pyplot as plt

def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:', x.shape, y.shape)
"""
np.meshgrid()是NumPy库中的一个函数，用于生成网格坐标矩阵。它接受一维数组作为输入，并产生对应的多维坐标矩阵。
np.meshgrid()的语法如下：
    X1, X2, ... = np.meshgrid(x1, x2, ...)
其中，x1, x2, ...是一维数组，表示在不同维度上的坐标值。
np.meshgrid()的返回值是与输入数组个数相等的多维数组，每个数组的形状都与输入数组相同。这些数组可以用来表示坐标矩阵的不同维度。
"""
X, Y = np.meshgrid(x, y)
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

x = torch.tensor([0., 0.], requires_grad=True)
"""
torch.optim.Adam()是PyTorch库中的一个优化器类，用于实现Adam优化算法。它是一种常用的梯度下降算法，可以用于更新神经网络中的参数。
torch.optim.Adam()的语法如下：
    optimizer = torch.optim.Adam(params, lr=1e-3)
其中，params是一个可迭代对象，包含了需要优化的模型参数。lr是学习率（learning rate），用于控制参数更新的步长。
创建torch.optim.Adam()实例后，你可以使用其提供的方法来更新参数。例如，可以使用optimizer.step()方法来执行参数更新，
使用optimizer.zero_grad()方法来清除梯度。
"""
optimizer = torch.optim.Adam([x], lr=1e-3)
for step in range(20000):
    pred = himmelblau(x)
    optimizer.zero_grad()
    pred.backward()
    optimizer.step()
    if step % 2000 == 0:
        # x.tolist() 是一个用于将 NumPy 数组 x 转换为 Python 列表的方法。
        print('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), pred.item()))