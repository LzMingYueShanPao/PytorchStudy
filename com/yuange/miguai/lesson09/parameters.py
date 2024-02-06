import torch.nn as nn
import torch.optim as optim

net = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 2))
# torch.Size([2, 4])
print(list(net.parameters())[0].shape)
# torch.Size([2])
print(list(net.parameters())[3].shape)
# Parameter containing:
# tensor([[-0.4374,  0.4693,  0.0243,  0.4533],
#         [ 0.3674,  0.1715,  0.0967,  0.2409]], requires_grad=True)
print(list(net.parameters())[0])
# Parameter containing:
# tensor([0.3114, 0.3310], requires_grad=True)
print(list(net.parameters())[1])
# dict_items([('0.weight', Parameter containing:
# tensor([[-0.4374,  0.4693,  0.0243,  0.4533],
#         [ 0.3674,  0.1715,  0.0967,  0.2409]], requires_grad=True)), ('0.bias', Parameter containing:
# tensor([0.3114, 0.3310], requires_grad=True)), ('1.weight', Parameter containing:
# tensor([[ 0.7048, -0.2759],
#         [ 0.1749,  0.6400]], requires_grad=True)), ('1.bias', Parameter containing:
# tensor([-0.3198, -0.0600], requires_grad=True))])
print(dict(net.named_parameters()).items())

optimizer = optim.SGD(net.parameters(), lr=1e-3)
# SGD (
# Parameter Group 0
#     dampening: 0
#     differentiable: False
#     foreach: None
#     lr: 0.001
#     maximize: False
#     momentum: 0
#     nesterov: False
#     weight_decay: 0
# )
print(optimizer)
