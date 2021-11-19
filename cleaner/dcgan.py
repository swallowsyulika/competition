import torch
import torch.nn as nn

nf = 64

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # 1 x 128 x 128
            nn.Conv2d(1, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True),
            # nf x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
            # 2nf x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            # 4nf x 16 x 16 
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            # 8nf x 8 x 8
            nn.Conv2d(nf * 8, nf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 16),
            nn.LeakyReLU(0.2, True),
            # 16nf x 4 x 4
            nn.ConvTranspose2d(nf * 16, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            # 8nf x 8 x 8
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            # 4nf x 16 x 16
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
            # 2nf x 32 x 32
            nn.ConvTranspose2d(nf * 2, nf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 1),
            nn.LeakyReLU(0.2, True),
            # nf x 64 x 64
            nn.ConvTranspose2d(nf, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 1 x 128 x 128
        )
    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # 1 x 128 x 128
            nn.Conv2d(2, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True),
            # nf x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
            # 2nf x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            # 4nf x 16 x 16 
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            # 8nf x 8 x 8
            nn.Conv2d(nf * 8, nf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 16),
            nn.LeakyReLU(0.2, True),
            # 16nf x 4 x 4
            nn.Conv2d(nf * 16, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid(),
            # 1 x 1 x 1
        )
    
    def forward(self, dirty_imgs, clean_imgs):
        
        x = torch.cat((dirty_imgs, clean_imgs), dim=1)
        
        return self.main(x)

if __name__ == "__main__":

    dummy_img = torch.zeros((1, 1, 128, 128))
    d = Generator()
    out = d(dummy_img)
    print(out.shape)