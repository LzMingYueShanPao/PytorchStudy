import torch
from torch.nn import functional as F

x = torch.ones(1)
w = torch.full([1], 2.)
mse = F.mse_loss(torch.ones(1), x*w)
print(mse)

# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
# torch.autograd.grad(mse, [w])

w.requires_grad_()

# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
# torch.autograd.grad(mse, [w])

mse = F.mse_loss(torch.ones(1), x*w)
"""
mse.backward()是PyTorch中的一个方法，用于计算相对于某个张量（通常是损失函数）的梯度，并将梯度值保存在相应的张量对象中。
"""
mse.backward()
print(w.grad)