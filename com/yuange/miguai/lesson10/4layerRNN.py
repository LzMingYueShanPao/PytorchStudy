import torch.nn as nn
import torch

rnn = nn.RNN(100, 20, num_layers=4)
# RNN(100, 20, num_layers=4)
print(rnn)
x = torch.randn(10, 3, 100)
out, h = rnn(x)
# torch.Size([10, 3, 20]) torch.Size([4, 3, 20])
print(out.shape, h.shape)