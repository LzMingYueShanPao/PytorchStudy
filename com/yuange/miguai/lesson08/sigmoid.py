import torch

a = torch.linspace(-100, 100, 10)
# tensor([-100.0000,  -77.7778,  -55.5556,  -33.3333,  -11.1111,   11.1111,
#           33.3333,   55.5556,   77.7778,  100.0000])
print(a)
# tensor([0.0000e+00, 1.6655e-34, 7.4564e-25, 3.3382e-15, 1.4945e-05, 9.9999e-01,
#         1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00])
print(torch.sigmoid(a))