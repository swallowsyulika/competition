from cv2 import Stitcher
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

# class Net(nn.Module):

#     def __init__(self, num_features = 3000, nf = 32):
#         super().__init__()

#         self.main = nn.Sequential(
#             # 1 x 128 x 128
#             nn.Conv2d(1, nf, 4, 2, 1),
#             nn.InstanceNorm2d(nf),
#             nn.ReLU(True),

#             # nf * 64 x 64
#             nn.Conv2d(nf, nf * 2, 4, 2, 1),
#             nn.InstanceNorm2d(nf * 2),
#             nn.ReLU(True),

#             # 2nf * 32 * 32
#             nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
#             nn.InstanceNorm2d(nf * 4),
#             nn.ReLU(True),

#             # 4nf * 16 * 16
#             nn.Conv2d(nf * 4 , nf * 8, 4, 2, 1),
#             nn.InstanceNorm2d(nf * 8),
#             nn.ReLU(True),
#             nn.Dropout(0,5),

#             # 8nf * 8 * 8
#             nn.Conv2d(nf * 8 , nf * 16, 4, 2, 1),
#             nn.InstanceNorm2d(nf * 16),
#             nn.ReLU(True),
#             nn.Dropout(0.5),
#             nn.Flatten(),
#             # 16nf * 4 * 4
#             nn.Linear(nf * 16 * 4 * 4, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 128),
#             nn.ReLU(True),
#             nn.Linear(128, num_features)
#         )
#     def forward(self, x):

#         return self.main(x)

# class Net(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(7744, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = F.dropout(x, p=0.7)
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
        
#         return x
# class Net(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 6, 3, 2)
#         self.conv2 = nn.Conv2d(6, 16, 3, 2)
#         self.fc1 = nn.Linear(16 * 15 * 15, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         # print(x.shape)
#         x = F.dropout(x, p=0.1)
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
        
#         return x


# v2 architecture
# class Net(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.main = nn.Sequential(
#             # 1 x 64 x 64
#             nn.Conv2d(1, 8, 4, 2, 1),
#             nn.InstanceNorm2d(8),
#             nn.LeakyReLU(0.2),
#             # 4 x 32 x 32
#             nn.Conv2d(8, 16, 4, 2, 1),
#             nn.InstanceNorm2d(16),
#             nn.LeakyReLU(0.2),
#             # 16 x 16 x 16
#             nn.Conv2d(16, 32, 4, 2, 1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2),
#             # 32 x 8 x 8
#             nn.Conv2d(32, 512, 4, 2, 1),
#             nn.InstanceNorm2d(512),
#             nn.LeakyReLU(0.2),
#             # 64 x 4 x 4
#             nn.Flatten(),
#             nn.Linear(512 * 4 * 4, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, num_classes)
#         )

#     def forward(self, x):
#         return self.main(x)
# v3 architecture

# class ResBlock(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super(ResBlock, self).__init__()
        
#         self.main = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
#             nn.InstanceNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
#             nn.InstanceNorm2d(out_channels),
#         )        

#     def forward(self, x):
#         return x + self.main(x)

# class Net(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.down_blocks = nn.Sequential(
#             # 1 x 64 x 64
#             nn.Conv2d(1, 4, 4, 2, 1, bias=False),
#             nn.InstanceNorm2d(4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 4 x 32 x 32
#             nn.Conv2d(4, 16, 4, 2, 1, bias=False),
#             nn.InstanceNorm2d(16),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 16 x 16 x 16
#             nn.Conv2d(16, 32, 4, 2, 1, bias=False),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 32 x 8 x 8
#             nn.Conv2d(32, 64, 4, 2, 1, bias=False),
#             nn.InstanceNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 64 x 4 x 4
#         )
#         self.residuals = nn.Sequential(*[ResBlock(64, 64) for _ in range(8)])

#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 4 * 4, 512),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(512, 512),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):

#         x = self.down_blocks(x)
#         x = self.residuals(x)
#         x = self.fc(x)

#         return x

# v3 (FE)
# class Net(nn.Module):

#     def __init__(self, num_classes: int):
#         super().__init__()

#         self.pool = nn.MaxPool2d((2, 2))
#         self.relu = nn.ReLU(True)

#         # ConvBlock 1
#         self.conv_1_1 = nn.Conv2d(1, 4, 3, 1, 1)
#         self.conv_1_2 = nn.Conv2d(4, 4, 3, 1, 1)
#         self.conv_1_3 = nn.Conv2d(4, 8, 3, 1, 1)
#         self.conv_1_bn = nn.BatchNorm2d(8)

#         # ConvBlock 2
#         self.conv_2_1 = nn.Conv2d(8, 16, 3, 1, 1)
#         self.conv_2_2 = nn.Conv2d(16, 16, 3, 1, 1)
#         self.conv_2_3 = nn.Conv2d(16, 32, 3, 1, 1)
#         self.conv_2_bn = nn.BatchNorm2d(32)

#         # ConvBlock 3
#         self.conv_3_1 = nn.Conv2d(32, 64, 3, 1, 1)
#         self.conv_3_2 = nn.Conv2d(64, 64, 3, 1, 1)
#         self.conv_3_3 = nn.Conv2d(64, 128, 3, 1, 1)
#         self.conv_3_bn = nn.BatchNorm2d(128)

#         # ConvBlock 4
#         self.conv_4_1 = nn.Conv2d(128, 512, 3, 1, 1)
#         self.conv_4_2 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.conv_4_3 = nn.Conv2d(512, 1024, 3, 1, 1)
#         self.conv_4_bn = nn.BatchNorm2d(1024)

#         # ConvBlock 5
#         self.conv_5_1 = nn.Conv2d(1024, 512, 3, 1, 1)
#         self.conv_5_2 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.conv_5_3 = nn.Conv2d(512, 128, 3, 1, 1)
#         self.conv_5_bn = nn.BatchNorm2d(128)

#         self.fc1 = nn.Linear(128 * 2 * 2, 1024)
#         self.fc2 = nn.Linear(1024, 1024)
#         self.fc3 = nn.Linear(1024, num_classes)

#     def forward(self, x):
#         # 1 x 64 x 64

#         # ConvBlock 1
#         x = self.relu(self.conv_1_1(x))
#         x = self.relu(self.conv_1_2(x))
#         x = self.relu(self.conv_1_3(x))

#         x = self.conv_1_bn(self.pool(x))
#         # 16 x 32 x 32

#         # ConvBlock 2
#         x = self.relu(self.conv_2_1(x))
#         x = self.relu(self.conv_2_2(x))
#         x = self.relu(self.conv_2_3(x))

#         x = self.conv_2_bn(self.pool(x))
#         # 64 x 16 x 16

#         # ConvBlock 3
#         x = self.relu(self.conv_3_1(x))
#         x = self.relu(self.conv_3_2(x))
#         x = self.relu(self.conv_3_3(x))

#         x = self.conv_3_bn(self.pool(x))
#         # 64 x 8 x 8

#         # ConvBlock 4
#         x = self.relu(self.conv_4_1(x))
#         x = self.relu(self.conv_4_2(x))
#         x = self.relu(self.conv_4_3(x))

#         x = self.conv_4_bn(self.pool(x))

#         # x 4 x 4

#         # ConvBlock 5
#         x = self.relu(self.conv_5_1(x))
#         x = self.relu(self.conv_5_2(x))
#         x = self.relu(self.conv_5_3(x))

#         x = self.conv_5_bn(self.pool(x))
#         # 4096 x 2 x 2
#         x = torch.flatten(x, start_dim=1)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)

#         return x

from torchvision import models

class Net(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()

        self.backbone = models.resnet152(pretrained=True)
        self.fc1 = nn.Linear(1000, num_classes)

    
    def forward(self, x):

        # b_size x 1 x 64 x 64
        x = x.repeat(1, 3, 1, 1)

        x = self.backbone(x)
        x = self.fc1(x)

        return x

