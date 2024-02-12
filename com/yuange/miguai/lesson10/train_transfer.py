import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader

from com.yuange.miguai.lesson10.pokemon import Pokemon
# from com.yuange.miguai.lesson10.resnet import ResNet18
from torchvision.models import resnet18
from com.yuange.miguai.lesson10.utils import Flatten


batchsz = 32
lr = 1e-3
epochs = 10

device = torch.device('cuda')
# 设置随机种子可以保证每次训练得到的结果是确定性的，也就是说，无论何时运行代码，生成的随机数序列都是相同的。
torch.manual_seed(1234)

train_db = Pokemon('pokemon', 224, mode='train')
val_db = Pokemon('pokemon', 224, mode='val')
test_db = Pokemon('pokemon', 224, mode='test')
# num_workers=4：表示用于数据加载的子进程数量。
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)

viz = visdom.Visdom()

def evalute(model, loader):
    """
    将模型设置为评估（inference）模式。
    在评估模式下，模型中的 Batch Normalization 层会使用移动平均的统计数据而非当前 mini-batch 的统计数据来进行归一化，这有助于提高模型的泛化能力。
    另外，Dropout 层会被关闭，即不再随机丢弃神经元，而是保留所有神经元进行前向传播。
    """
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x,y in loader:
        x, y = x.to(device), y.to(device)
        # 关闭梯度计算
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total

def main():
    # model = ResNet18(5).to(device)
    """
    创建一个预训练的 ResNet-18 模型，并加载在 ImageNet 数据集上预训练的权重。
    resnet18 是 PyTorch 中提供的一个函数，用于实例化一个 ResNet-18 模型。
    """
    train_model = resnet18(pretrained=True)
    model = nn.Sequential(
        *list(train_model.children())[:-1], # 将 train_model 中的所有子模块（即网络的各个层）转换为列表，然后去掉原始 ResNet-18 模型中的最后一层全连接层（通常是用于分类的层）。
        Flatten(), # 自定义的一个展平层，用于将卷积层输出的多维特征图展平成一维向量，为接下来的全连接层做准备。
        nn.Linear(512, 5) # 添加一个全连接层，将 ResNet-18 最后一个卷积层的输出（大小为 512）映射到输出类别数为 5 的空间，这里假设是一个 5 类分类问题。
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            """
            将模型设置为训练模式。
            在训练模式下，Batch Normalization 层会使用当前 mini-batch 的统计数据来进行归一化，而不是移动平均的统计数据；
            Dropout 层会随机丢弃部分神经元，以防止过拟合。
            """
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epoch % 1 == 0:
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best.mdl')
                viz.line([val_acc], [global_step], win='val_acc', update='append')

    print('best acc:', best_acc, 'best epoch:', best_epoch)
    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')
    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)

if __name__ == '__main__':
    main()