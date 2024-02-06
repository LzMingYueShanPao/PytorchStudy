import torch

a = torch.rand(4,1,28,28)
# torch.Size([4, 1, 28, 28])
print(a.shape)
"""
view() 是 PyTorch 张量的一个方法，用于改变张量的形状（即维度）。它可以用来调整张量的大小，但要注意调整后的大小必须与原始张量中元素的总数相匹配。
view() 方法的使用方式如下：
    new_tensor = tensor.view(*shape)
其中，tensor 是要进行形状调整的原始张量，*shape 是一个可变参数，用于指定调整后的形状。
以下是 view() 方法的一些常见用法和注意事项：
    view() 并不会改变张量中的数据，它只是返回一个具有新形状的新张量。如果你希望在改变形状的同时也改变数据，可以使用 reshape() 方法。
    调整后的形状的维度大小需要与原始张量的元素总数相匹配，否则会抛出错误。PyTorch 会自动计算缺失的维度大小，只需将其设置为 -1 即可。
    view() 方法返回的张量与原始张量共享内存，也就是说它们在内存中引用相同的数据。因此，在修改其中一个张量时，另一个张量也会受到影响。
    如果你不确定应该使用哪个具体的形状参数，可以使用 -1 表示自动计算。例如，tensor.view(-1) 可以将任意形状的张量展平为一维。
"""
# tensor([[0.8188, 0.8142, 0.9168,  ..., 0.7044, 0.1162, 0.7897],
#         [0.1104, 0.2141, 0.4557,  ..., 0.6482, 0.4336, 0.9149],
#         [0.6386, 0.3636, 0.2975,  ..., 0.4600, 0.8401, 0.2522],
#         [0.3045, 0.8690, 0.7650,  ..., 0.3806, 0.5809, 0.3819]])
print(a.view(4, 28 * 28))
# torch.Size([4, 784])
print(a.view(4, 28 * 28).shape)
# torch.Size([112, 28])
print(a.view(4 * 28, 28).shape)
# torch.Size([4, 28, 28])
print(a.view(4 * 1, 28, 28).shape)
b = a.view(4, 784)
# tensor([[0.8188, 0.8142, 0.9168,  ..., 0.7044, 0.1162, 0.7897],
#         [0.1104, 0.2141, 0.4557,  ..., 0.6482, 0.4336, 0.9149],
#         [0.6386, 0.3636, 0.2975,  ..., 0.4600, 0.8401, 0.2522],
#         [0.3045, 0.8690, 0.7650,  ..., 0.3806, 0.5809, 0.3819]])
print(b)
print(b.view(4, 28, 28, 1).shape)
# RuntimeError: shape '[4, 783]' is invalid for input of size 3136
print(a.view(4,783))