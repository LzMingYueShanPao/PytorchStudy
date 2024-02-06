import torch
from torch.nn import functional as F

a = torch.rand(3)
"""
requires_grad_()是PyTorch中的一个方法，用于原地修改张量的 requires_grad 属性。设置 requires_grad=True 可以告诉 PyTorch 跟踪此张量的梯度，从而可以通过自动微分计算相应的梯度值。
当我们定义一个张量时，默认情况下它的 requires_grad 属性为 False。通过调用 requires_grad_() 方法并传递 True 或 False 作为参数，可以改变张量是否需要梯度跟踪。
"""
# tensor([0.6832, 0.7874, 0.2356], requires_grad=True)
print(a.requires_grad_())

"""
F.softmax() 是 PyTorch 中的一个函数，用于将张量转换为概率分布。该函数将输入的张量作为参数，并对其进行指数化（exponential），然后对结果进行归一化（normalize），从而得到一个概率分布。
"""
p = F.softmax(a, dim=0)
# tensor([0.3638, 0.4037, 0.2325], grad_fn=<SoftmaxBackward0>)
print(p)

# RuntimeError: grad can be implicitly created only for scalar outputs
# p.backward()

"""
p[1]表示第2个梯度
p[2]表示第3个梯度
"""
# tensor(0.4037, grad_fn=<SelectBackward0>)
print(p[1])
# tensor(0.2325, grad_fn=<SelectBackward0>)
print(p[2])

"""
torch.autograd.grad() 是 PyTorch 中的一个函数，用于计算某个标量值相对于给定张量的梯度。该函数接受两个参数：
    第一个参数是要计算梯度的标量张量，
    第二个参数是要计算梯度的源张量列表。
函数返回的是一个张量列表，表示标量相对于每个源张量的梯度。
在您提供的代码中，p[1] 是一个标量张量，[a] 是一个张量列表，其中包含一个源张量 a。函数调用 torch.autograd.grad(p[1], [a], retain_graph=True) 
将计算标量 p[1] 相对于张量 a 的梯度，并在计算后保留计算图（retain_graph=True）。
"""
# (tensor([-0.0721,  0.1489, -0.0767]),)
print(torch.autograd.grad(p[1], [a], retain_graph=True))

# (tensor([-0.1671, -0.0767,  0.2439]),)
print(torch.autograd.grad(p[2], [a]))