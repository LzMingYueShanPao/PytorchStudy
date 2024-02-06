from torchvision import transforms, datasets
import torch
from torch.utils.data import DataLoader

def main():
    batchsz = 32

    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = next(iter(cifar_train))
    # x: torch.Size([32, 3, 32, 32]) label: torch.Size([32])
    print('x:', x.shape, 'label:', label.shape)

    # for epoch in range(1000):


if __name__ == '__main__':
    main()