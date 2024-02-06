import torch
from torch.nn import functional as F

x = torch.ones(1)
# tensor([1.])
print(x)
w = torch.full([1], 2.)
# tensor([2.])
print(w)
"""
F.mse_loss()是PyTorch中计算均方误差（Mean Squared Error，MSE）损失函数的函数。该函数将两个输入张量输入并计算它们之间的MSE损失。
mse_loss()函数有两个输入参数：input和target。其中，input表示模型输出的预测结果，target表示真实标签。函数返回input和target之间的MSE损失值。
"""
mse = F.mse_loss(torch.ones(1), x*w)
# tensor(1.)
print(mse)

"""
torch.autograd.grad()函数是PyTorch中用于计算梯度的函数。它接受一个标量值（如损失）和一组变量（如权重），并返回相对于这些变量的梯度。
"""
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
# RuntimeError：张量的元素0不需要grad，也没有grad_fn
# torch.autograd.grad(mse, [w])

"""
w.requires_grad_()是PyTorch中的一个方法，用于原地修改张量 w 的 requires_grad 属性。通过调用 requires_grad_() 方法并传递 True 或 False 作为参数，可以改变张量是否需要梯度跟踪。
"""
# tensor([2.], requires_grad=True)
print(w.requires_grad_())
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
# torch.aotograd.grad(mse, [w])
mse = F.mse_loss(torch.ones(1), x*w)
torch.autograd.grad(mse, [w])
