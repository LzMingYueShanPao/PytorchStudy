import torch
import torch.nn.functional as F

x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)

o = torch.sigmoid(x@w.t())
# torch.Size([1, 2])
print(o.shape)

loss = F.mse_loss(torch.ones(1,2), o)
# tensor(0.3365, grad_fn=<MseLossBackward0>)
print(loss)

loss.backward()

# tensor([[ 0.0359, -0.1059, -0.0580,  0.0578,  0.1533,  0.0407, -0.0764, -0.0961,
#           0.0571, -0.1299],
#         [ 0.0571, -0.1681, -0.0920,  0.0918,  0.2435,  0.0647, -0.1214, -0.1527,
#           0.0906, -0.2064]])
print(w.grad)