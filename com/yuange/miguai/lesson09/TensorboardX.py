from tensorboardX import SummaryWriter
import torch

# 创建一个写入器（writer）对象，指定保存日志文件的目录
writer = SummaryWriter('logs')

for i in range(10):
    # 模拟训练过程中的损失函数值
    loss = 0.1 * i
    # 使用 writer 记录损失函数值和对应的步数
    writer.add_scalar('Loss/train', loss, i)

weights = torch.randn(100, 10)
# 记录直方图（Histograms）
writer.add_histogram('Weights', weights)

writer.close()