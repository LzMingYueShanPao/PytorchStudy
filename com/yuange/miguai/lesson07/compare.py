import torch

a = torch.randn(4, 10)
# tensor([[False,  True, False, False,  True,  True, False, False, False,  True],
#         [False,  True,  True, False,  True, False, False,  True,  True,  True],
#         [False,  True, False, False, False, False,  True, False,  True, False],
#         [ True,  True,  True, False,  True,  True, False, False,  True,  True]])
print(a > 0)
"""
在 PyTorch 中，torch.gt() 是一个函数，用于执行元素级别的大于（greater than）比较操作。它将比较两个张量中对应位置的元素，并返回一个新的布尔张量，指示对应位置上的元素是否满足大于关系。
"""
# tensor([[False,  True, False, False,  True,  True, False, False, False,  True],
#         [False,  True,  True, False,  True, False, False,  True,  True,  True],
#         [False,  True, False, False, False, False,  True, False,  True, False],
#         [ True,  True,  True, False,  True,  True, False, False,  True,  True]])
print(torch.gt(a, 0))
# tensor([[True, True, True, True, True, True, True, True, True, True],
#         [True, True, True, True, True, True, True, True, True, True],
#         [True, True, True, True, True, True, True, True, True, True],
#         [True, True, True, True, True, True, True, True, True, True]])
print(a != 0)
a = torch.ones(2, 3)
b = torch.randn(2, 3)
"""
在 PyTorch 中，torch.eq() 是一个函数，用于执行元素级别的相等（equal）比较操作。它将比较两个张量中对应位置的元素，并返回一个新的布尔张量，指示对应位置上的元素是否满足相等关系。
torch.eq() 函数的使用方式如下：
    output = torch.eq(input, other)
其中，input 和 other 是要进行比较的两个张量。input 是主张量，而 other 是第二个张量或标量值。input 和 other 必须具有相同的形状或可以进行广播（broadcasting）。
函数将返回一个与 input 相同形状的布尔张量 output，其中的每个元素都表示对应位置的元素是否满足相等关系。
"""
# tensor([[False, False, False],
#         [False, False, False]])
print(torch.eq(a, b))
# tensor([[True, True, True],
#         [True, True, True]])
print(torch.eq(a, a))
"""
在PyTorch中，torch.equal()是一个函数，用于判断两个张量是否相等。
torch.equal()函数的使用方式如下：
    torch.equal(input, other)
其中，input和other是要进行比较的两个张量。函数将返回一个布尔值，表示两个张量是否相等。如果两个张量具有相同的形状和元素值，则返回True，否则返回False。
"""
# True
print(torch.equal(a, a))
"""
torch.equal()和torch.eq()都是用于比较两个张量是否相等的函数，但它们之间有一些区别。
主要区别如下：
    torch.equal()函数比较两个张量在形状和元素值上是否完全相同，如果相同返回True，否则返回False。
    torch.eq()函数执行元素级别的相等（equal）比较操作。它将比较两个张量中对应位置的元素，并返回一个新的布尔张量，指示对应位置上的元素是否满足相等关系。
    torch.equal()函数返回一个布尔值，而torch.eq()函数返回一个布尔张量。
"""