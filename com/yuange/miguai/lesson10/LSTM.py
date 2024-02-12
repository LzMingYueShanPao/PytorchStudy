import torch
import torch.nn as nn

lstm = nn.LSTM(input_size=100, hidden_size=20, num_layers=4)
# LSTM(100, 20, num_layers=4)
print(lstm)
x = torch.randn(10, 3, 100)
out, (h, c) = lstm(x)
# torch.Size([10, 3, 20])
# torch.Size([4, 3, 20])
# torch.Size([4, 3, 20])
print(out.shape, h.shape, c.shape)