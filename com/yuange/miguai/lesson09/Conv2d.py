import torch
import torch.nn as nn

layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)
x = torch.rand(1, 1, 28, 28)

out = layer.forward(x)
# torch.Size([1, 3, 26, 26])
print(out.shape)

layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
out = layer.forward(x)
# torch.Size([1, 3, 28, 28])
print(out.shape)

layer = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1)
out = layer.forward(x)
# torch.Size([1, 3, 14, 14])
print(out.shape)

out = layer(x)
# torch.Size([1, 3, 14, 14])
print(out.shape)

# tensor([[[[-0.0304, -0.2558, -0.2640],
#           [-0.1062, -0.2596,  0.2215],
#           [ 0.0684, -0.2321,  0.1789]]],
#
#
#         [[[ 0.2224,  0.0571,  0.0023],
#           [ 0.3032,  0.0216,  0.2383],
#           [ 0.2195,  0.2115, -0.0531]]],
#
#
#         [[[-0.2508, -0.1882, -0.1885],
#           [-0.0742,  0.2146, -0.2081],
#           [ 0.0371, -0.2444,  0.0885]]]], requires_grad=True)
print(layer.weight)
# torch.Size([3, 1, 3, 3])
print(layer.weight.shape)
# torch.Size([3])
print(layer.bias.shape)