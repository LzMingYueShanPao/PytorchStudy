import torch.nn as nn
import torch

rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=1)
# RNN(100, 20)
print(rnn)
x = torch.randn(10, 3, 100)
"""
第一个参数 x 指的是输入特征的维度。
第二个参数 torch.zeros(1, 3, 20) 指的是隐藏层的特征维度。
"""
out, h = rnn(x, torch.zeros(1, 3, 20))
# torch.Size([10, 3, 20]) torch.Size([1, 3, 20])
print(out.shape, h.shape)