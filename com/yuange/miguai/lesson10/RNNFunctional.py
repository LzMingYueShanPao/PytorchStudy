import torch
import torch.nn as nn

x = torch.randn(10, 3, 100)
cell1 = nn.RNNCell(100, 30)
cell2 = nn.RNNCell(30, 20)
h1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
for xt in x:
    h1 = cell1(xt, h1)
    h12 = cell2(h1, h2)
# torch.Size([3, 20])
print(h2.shape)