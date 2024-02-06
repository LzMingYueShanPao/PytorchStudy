import torch

a = torch.tensor(3.14)
# tensor(3.1400)
print(a)
# floor() 是一个函数，用于返回不大于输入参数的最大整数。用于向下取整
# tensor(3.)
print(a.floor())
# ceil() 是一个函数，用于返回不小于输入参数的最小整数。用于向上取整
# tensor(4.)
print(a.ceil())
# trunc() 是一个函数，用于返回输入参数的整数部分。用于截断操作
# tensor(3.)
print(a.trunc())
# frac() 是一个函数或方法，用于获取数值的小数部分。
# tensor(0.1400)
print(a.frac())
a = torch.tensor(3.479)
# round() 是一个函数，用于将输入参数四舍五入到指定的小数位数。用于四舍五入操作
# tensor(3.4800)
print(a.round(decimals=2))
a = torch.tensor(3.5)
# tensor(4.)
print(a.round())