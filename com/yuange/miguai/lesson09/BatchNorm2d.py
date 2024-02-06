import torch
import torch.nn as nn

x = torch.rand(1, 16, 7, 7)
layer = nn.BatchNorm2d(16)

out = layer(x)
# torch.Size([1, 16, 7, 7])
print(out.shape)
# Parameter containing:
# tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#        requires_grad=True)
print(layer.weight)
# torch.Size([16])
print(layer.weight.shape)
# torch.Size([16])
print(layer.bias.shape)
# print(vars(layer))

# BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
print(layer.eval())
out = layer(x)
# torch.Size([1, 16, 7, 7])
print(out.shape)
print(vars(layer))
