import torch
import os, glob
import random, csv
import visdom
import time
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Pokemon(Dataset):
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        # 图片根目录
        self.root = root
        # 图片固定大小
        self.resize = resize

        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            # 给每一类图片设置序号标签，方便后续进行分类任务
            self.name2label[name] = len(self.name2label.keys())
        # {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}
        # print(self.name2label)

        self.images, self.labels = self.load_csv('images.csv')
        if mode == 'train':
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.6 * len(self.images)) : int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)) : int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            # 1167  ['pokemon\\bulbasaur\\00000000.png', 'pokemon\\bulbasaur\\00000001.png',...]
            print(len(images), images)
            # 随机打乱列表 images 中元素的顺序
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    # 对字符串 img 进行以操作系统路径分隔符为分隔符进行分割，并返回分割后的列表中倒数第二个元素。
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # pokemon\bulbasaur\00000000.png,0
                    writer.writerow([img, label])
                print('writen into csv file:', filename)
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)
        # 判断 images 列表和 labels 列表的长度是否相等，如果不相等则会引发一个 AssertionError 异常
        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        """
        torch.tensor(mean) 将列表或数组 mean 转换为一个 PyTorch 张量。
        .unsqueeze(1) 在第一个维度上插入一个新维度，即在原始张量的形状中添加一个长度为 1 的维度。
        结果是将原始张量从形状为 (a, b, c) 扩展为 (1, a, b, c)。
        .unsqueeze(1) 再次在第二个维度上插入一个新维度，同样在原始张量的形状中添加一个长度为 1 的维度。
        这样就将原始张量从形状为 (1, a, b, c) 扩展为 (1, 1, a, b, c)。
        """
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # 将图片路径转化为图片数据
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))), # 图片伸缩
            transforms.RandomRotation(15), # 将图片旋转15度角
            transforms.CenterCrop(self.resize), # 将图片裁剪为resize固定大小的正方形
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label

def main():
    viz = visdom.Visdom()

    # 第一个参数图片所处的目录路径
    db = Pokemon('pokemon', 64, 'train')

    x, y = next(iter(db))
    print('sample:', x.shape, y.shape, y)

    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)
    for x, y in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        time.sleep(10)

if __name__ == '__main__':
    main()