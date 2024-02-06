from torchvision import datasets,transforms
import torch

batch_size=200

"""
transforms.RandomHorizontalFlip()以一定的概率随机地水平翻转图像。
transforms.RandomVerticalFlip()以一定的概率随机地垂直翻转图像。
"""
"""
transforms.RandomRotation(15) 以一定的角度范围内随机旋转图像。此处以15度为例
transforms.RandomRotation([90, 180, 270]) 以90/180/270角度随机选一个进行旋转
"""
"""
transforms.Resize([32, 32]) 将图像调整为指定的大小 (32, 32)
"""
"""
transforms.RandomCrop([28, 28])将 [28, 28] 作为参数传递给 transforms.RandomCrop() 函数，以指定裁剪的目标大小为 28×28 像素。
"""
angles = [90, 180, 270]
random_choice_rotation = transforms.RandomChoice([
    transforms.RandomRotation(angle) for angle in angles
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomVerticalFlip(),
                       transforms.RandomRotation(15),
                       random_choice_rotation,
                       transforms.Resize([32, 32]),
                       transforms.RandomCrop([28, 28]),
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=True
)

"""
transforms.Resize((32, 32))将输入图像调整（缩放）到指定的大小，这里是 32x32 像素。
transforms.ToTensor() 将 PIL 图像或者 NumPy 数组转换为 FloatTensor，并且把图像的像素值范围从 [0, 255] 缩放到 [0.0, 1.0]。这是准备图像数据以供 PyTorch 模型使用时常见的一个步骤。
"""
cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
]),download=True)
cifar_train = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size, shuffle=True)



cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
]), download=True)