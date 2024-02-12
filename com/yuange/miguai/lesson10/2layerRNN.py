import torch.nn as nn

rnn = nn.RNN(100, 10, num_layers=2)
# odict_keys(['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0',
# 'weight_ih_l1', 'weight_hh_l1', 'bias_ih_l1', 'bias_hh_l1'])
print(rnn._parameters.keys())
# torch.Size([10, 10]) torch.Size([10, 100])
print(rnn.weight_hh_l0.shape, rnn.weight_ih_l0.shape)
# torch.Size([10, 10]) torch.Size([10, 10])
print(rnn.weight_hh_l1.shape, rnn.weight_ih_l1.shape)

