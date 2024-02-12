from torch import nn

"""
创建一个循环神经网络（Recurrent Neural Network, RNN）层。
第一个参数 (100) 指的是输入特征的维度。也就是说，每个输入序列的每个元素应该是一个100维的向量。
第二个参数 (10) 指的是隐藏层的特征维度。也就是说，RNN层内部每个时间步骤上的隐藏状态将会被映射为一个10维的向量。
"""
rnn = nn.RNN(100, 10)
# odict_keys(['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'])
print(rnn._parameters.keys())

# torch.Size([10, 100]) torch.Size([10, 100])
print(rnn.weight_ih_l0.shape, rnn.weight_ih_l0.shape)
# torch.Size([10]) torch.Size([10])
print(rnn.bias_ih_l0.shape, rnn.bias_hh_l0.shape)