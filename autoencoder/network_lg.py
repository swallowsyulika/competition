import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))

        self.conv_1 = nn.Conv2d(1, 16, 3, 1, 1)

        self.conv_2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv_2_bn = nn.BatchNorm2d(32)

        self.conv_3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv_3_bn = nn.BatchNorm2d(64)

        self.conv_4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv_4_bn = nn.BatchNorm2d(128)

    def forward(self, x):

        x = F.relu(self.pool(self.conv_1(x)))
        # print(x.shape)

        x = F.relu(self.pool(self.conv_2_bn(self.conv_2(x))))
        # print(x.shape)

        x = F.relu(self.pool(self.conv_3_bn(self.conv_3(x))))

        x = F.relu(self.pool(self.conv_4_bn(self.conv_4(x))))

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_conv_1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        
        self.t_conv_2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)

        self.t_conv_3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)

        self.t_conv_4 = nn.ConvTranspose2d(16, 1, 4, 2, 1)

    def forward(self, x):

        x = F.relu(self.t_conv_1(x))
        x = F.relu(self.t_conv_2(x))
        x = F.relu(self.t_conv_3(x))
        x = F.sigmoid(self.t_conv_4(x))

        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x