import torch

a = torch.rand(3, 4)
b = torch.rand(4)
# tensor([[0.8468, 0.4812, 0.1001, 0.1486],
#         [0.1483, 0.4215, 0.7973, 0.7360],
#         [0.0504, 0.9979, 0.0407, 0.6529]])
print(a)
# tensor([0.8926, 0.8779, 0.1250, 0.6849])
print(b)
# tensor([[1.7394, 1.3591, 0.2251, 0.8335],
#         [1.0409, 1.2994, 0.9222, 1.4208],
#         [0.9429, 1.8757, 0.1656, 1.3377]])
print(a+b)
# tensor([[1.7394, 1.3591, 0.2251, 0.8335],
#         [1.0409, 1.2994, 0.9222, 1.4208],
#         [0.9429, 1.8757, 0.1656, 1.3377]])
print(torch.add(a, b))
"""
torch.all(input) 是一个 PyTorch 函数，用于判断张量中的所有元素是否都为 True。如果所有元素都为 True，则返回一个标量 True；否则返回 False。
torch.all() 函数接受一个张量作为输入，并默认沿着所有维度进行操作。如果想要在指定维度上判断所有元素是否为 True，则可以使用 dim 参数来指定。
"""
# tensor(True)
print(torch.all(torch.eq(a - b, torch.sub(a, b))))
# tensor(True)
print(torch.all(torch.eq(a * b, torch.mul(a, b))))
# tensor(True)
print(torch.all(torch.eq(a / b, torch.div(a, b))))