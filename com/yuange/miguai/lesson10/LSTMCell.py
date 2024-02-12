import torch
import torch.nn as nn

"""
nn.LSTMCell
    基本单元：nn.LSTMCell代表了LSTM的一个单元，它仅处理序列中的一个时间步。如果你想要通过它处理整个序列，需要手动编写循环来依次传递每个时间步的数据。
    更多控制：使用nn.LSTMCell可以让你对LSTM的每个时间步有更细粒度的控制，比如你可以更灵活地实现一些复杂的循环神经网络结构，例如条件随机场（CRF）或者自定义的循环逻辑。
    手动状态管理：使用nn.LSTMCell时，你需要手动管理和初始化隐藏状态（h）和细胞状态（c），并在每个时间步之间传递这些状态。

nn.LSTM
    整个序列：与nn.LSTMCell不同，nn.LSTM是为处理整个序列而设计的。它自动处理序列的所有时间步，因此使用起来更简单，更直接。
    更少的控制，更高的便利性：nn.LSTM提供了一个更高层次的抽象，使得用户不需要编写时间步的循环，也不需要手动管理隐藏状态和细胞状态。这使得nn.LSTM更容易使用，特别是对于标准的序列处理任务。
    批处理优化：nn.LSTM内部实现了对整个序列的批处理优化，这通常比手动循环每个时间步使用nn.LSTMCell更高效。

总结
    如果你需要在序列处理中实现特定的、复杂的逻辑，或者想要对每个时间步有更细致的控制，那么nn.LSTMCell可能是更好的选择。
    如果你的任务是标准的序列处理，并且希望简化代码和提高效率，那么nn.LSTM会是更方便、更直接的选择。
"""
print('one layer lstm')
x = torch.randn(10, 3, 100)
cell = nn.LSTMCell(input_size=100, hidden_size=20)
h = torch.zeros(3, 20)
c = torch.zeros(3, 20)
for xt in x:
    h, c = cell(xt, [h, c])
# torch.Size([3, 20])
# torch.Size([3, 20])
print(h.shape, c.shape)

print('two layer lstm')
cell1 = nn.LSTMCell(input_size=100, hidden_size=30)
cell2 = nn.LSTMCell(input_size=30, hidden_size=20)
h1 = torch.zeros(3, 30)
c1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
c2 = torch.zeros(3, 20)
for xt in x:
    h1, c1 = cell1(xt, [h1, c1])
    h2, c2 = cell2(h1, [h2, c2])
# torch.Size([3, 20])
# torch.Size([3, 20])
print(h2.shape, c2.shape)