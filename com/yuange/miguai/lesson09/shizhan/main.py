from torchvision import transforms, datasets
import torch
from torch import optim,nn
from torch.utils.data import DataLoader

# from com.yuange.miguai.lesson09.shizhan.lenet5 import Lenet5
from com.yuange.miguai.lesson09.shizhan.resnet import ResNet18

def main():
    batchsz = 32

    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = next(iter(cifar_train))
    # x: torch.Size([32, 3, 32, 32]) label: torch.Size([32])
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    # model = Lenet5().to(device)
    model = ResNet18().to(device)
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range(1000):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32] [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits：[b, 10]   label：[b]
            loss = criteon(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss', loss.item())

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32] [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print(epoch, 'test acc', acc)



if __name__ == '__main__':
    main()