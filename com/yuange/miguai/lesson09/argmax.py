import torch
import torch.nn.functional as F

logist = torch.randn(4, 10)
pred = F.softmax(logist, dim=1)
# torch.Size([4, 10])
print(pred.shape)

pred_label = pred.argmax(dim=1)
# tensor([9, 4, 8, 7])
print(pred_label)
# tensor([9, 4, 8, 7])
print(logist.argmax(dim=1))

label = torch.tensor([9, 3, 2, 4])
correct = torch.eq(pred_label, label)
# tensor([ True, False, False, False])
print(correct)
# 0.25
print(correct.sum().float().item()/4)