import torch
import torch.nn.functional as F

x = torch.randn(1, 784)
w = torch.randn(10, 784)
logist = x@w.t()
# torch.Size([1, 10])
print(logist.shape)

pred = F.softmax(logist, dim=1)
# torch.Size([1, 10])
print(pred.shape)
pred_log = torch.log(pred)
# torch.Size([1, 10])
print(pred_log.shape)

"""
F.cross_entropy 是 PyTorch 中用于计算交叉熵损失的函数。
第一个参数是模型的输出。第二个参数是目标标签，每个元素表示对应样本的真实类别。
"""
# tensor(39.5020)
print(F.cross_entropy(logist, torch.tensor([3])))
# tensor(39.5020)
print(F.nll_loss(pred_log, torch.tensor([3])))