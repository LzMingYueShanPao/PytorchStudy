import torch
import torch.nn.functional as F

x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad=True)

o = torch.sigmoid(x@w.t())
# torch.Size([1, 1])
print(o.shape)

loss = F.mse_loss(torch.ones(1, 1), o)
# torch.Size([])
print(loss.shape)

loss.backward()

# tensor([[-0.1582,  0.0531,  0.1367,  0.3418, -0.0490,  0.0683, -0.0805, -0.2915,
#          -0.0244, -0.2084]])
print(w.grad)