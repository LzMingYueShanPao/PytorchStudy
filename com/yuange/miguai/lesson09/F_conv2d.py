import torch
import torch.nn.functional as F

x = torch.rand(1, 1, 28, 28)
w = torch.rand(16, 3, 5, 5)
b = torch.rand(16)

# RuntimeError: Given groups=1, weight of size [16, 3, 5, 5], expected input[1, 1, 28, 28] to have 3 channels,
# but got 1 channels instead（输入通道和权重通道数必须一致，否则报错）
# out = F.conv2d(x, w, b, stride=1, padding=1)

x = torch.randn(1, 3, 28, 28)
out = F.conv2d(x, w, b, stride=1, padding=1)
# torch.Size([1, 16, 26, 26])
print(out.shape)

out = F.conv2d(x, w, b, stride=2, padding=2)
# torch.Size([1, 16, 14, 14])
print(out.shape)