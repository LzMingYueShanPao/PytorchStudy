import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
        self.criteon = nn.MSELoss()

    def forward(self, x):
        batchsz = x.size(0)
        x = x.view(batchsz, 784)
        h_ = self.encoder(x)
        """
        chunk()函数是一种将数据分割成固定大小块的方法
        """
        # [b, 20] => [b, 10] and [b, 10]
        mu, sigma = h_.chunk(2, dim=1)
        """
        重参数化技巧
        """
        # reparametrize trick, epison~N(0, 1)
        h = mu + sigma * torch.randn_like(sigma)

        x_hat = self.decoder(h)
        x_hat = x_hat.view(batchsz, 1, 28, 28)

        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (batchsz * 28 * 28)
        return x_hat, kld