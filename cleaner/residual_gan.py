import torch
import torch.nn as nn

nf = 64

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.main(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: bool = True, normalization = True):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels) if normalization else nn.Identity(),
            nn.ReLU(True) if activation else nn.Identity()
        )
    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlock, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
        )        

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.down_blocks = nn.Sequential(
            # 1 x 128 x 128
            DownBlock(1, nf),
            # nf x 64 x 64
            DownBlock(nf, 2 * nf),
            # 2nf x 32 x 32
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(2 * nf, 2 * nf) for x in range(4)]
        )

        self.up_blocks = nn.Sequential(
            # 2nf x 32 x 32
            UpBlock(2 * nf, nf),
            # nf x 64 x 64
            UpBlock(nf, 1, activation = False, normalization = False),
            # 1 x 128 x 128

        )

    def forward(self, x):
        x = self.down_blocks(x)
        x = self.res_blocks(x)
        x = self.up_blocks(x)

        return torch.tanh(x)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # 2 x 128 x 128
            DownBlock(2, nf),
            # nf x 64 x 64
            DownBlock(nf, nf * 2),
            # 2nf x 32 x 32
            DownBlock(nf * 2, nf * 4),
            # 4nf x 16 x 16 
            DownBlock(nf * 4, nf * 8),
            # 8nf x 8 x 8
            DownBlock(nf * 8, nf * 16),
            # 16nf x 4 x 4
            nn.Conv2d(nf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # 1 x 1 x 1
        )
    
    def forward(self, dirty_imgs, clean_imgs):

        x = torch.cat((dirty_imgs, clean_imgs), dim=1)
        
        return self.main(x)