import json
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
from datasets import *
from custom_transforms import RandomDropResolution, RandomLine, RandomRatioPad
from utils import filter_character_list

with open("characters.json", 'r') as f:
    characters = json.loads(f.read())

print(characters)

class ColorChange:

    def __call__(self, img):
        ref = img.clone()

        background_color = random.uniform(0.0, 1.0)

        while True:
            text_color = random.uniform(0.0, 1.0)
            if abs(text_color - background_color) >= 0.05:
                break

        # print(background_color)
        # print(text_color)

        img[ref == 0] = text_color
        img[ref > 0] = background_color

        return img


class ThreeDimensionalize:

    def __init__(self, x_range, y_range) -> None:
        self.x_range = x_range
        self.y_range = y_range

    def __call__(self, img):

        offset_x = random.randrange(*self.x_range)
        offset_y = random.randrange(*self.y_range)

        background_color = random.uniform(0.0, 1.0)
        fore_text_color = random.uniform(0.0, 1.0)
        back_text_color = random.uniform(0.0, 1.0)

        inv_text_map = (img != 0).float()  # text -> 0, bg -> 1
        # draw background
        img = img * 0 + background_color

        # draw background text
        inv_bg_text_map = TF.affine(
            inv_text_map, angle=0, scale=1, shear=0, translate=(offset_x, offset_y), fill=1.0)

        img = img * inv_bg_text_map
        img[img == 0] = back_text_color

        # draw foreground text
        img = img * inv_text_map
        img[img == 0] = fore_text_color

        return img


class AddBorder:

    def __init__(self) -> None:
        self.kernel = np.ones((5, 5), np.uint8)

    def __call__(self, img):
        #ref = img.clone()

        #background_color = random.uniform(0.0, 1.0)
        dilation = cv2.erode(img.numpy(), self.kernel, iterations=1)

        return torch.from_numpy(dilation)
font_path = "fonts/KouzanGyousho.ttf"
filtered_characters = filter_character_list(font_path, characters)
generator = BackgroundGenerator("textures/", zones_range=(10, 30))
dirty_ds = TextureDBorderFontDataset(filtered_characters,
                       font_path,                       
                       font_size=48,
                       img_size=48,
                       border_size=(2, 2),
                       out_offset=(1, 2),
                       shadow_offset_x=(-2,2),
                       shadow_offset_y=(-2, 2),
                       #random_character_color=True,
                       bg_generator=generator,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #RandomRatioPad(),                           
                           transforms.Resize((128, 128)),                           
                           #transforms.RandomPerspective(0.5, 0.5, fill=0),
                           #transforms.RandomResizedCrop(128)
                           
                           
                           #RandomLine(128, (0, 5)),
                           #ThreeDimensionalize((-5, 5), (-5,5)),
                           #AddBorder(),
                       ]))
loader = DataLoader(dirty_ds, batch_size=2400, shuffle=False, num_workers=4)

img = dirty_ds[150]
print("showing")
#cv2.imshow("title", img[0].numpy() * 255)
# plt.imshow(img[0], cmap='gray', vmin=0, vmax=1)
# plt.show()

for idx, imgs in enumerate(loader):
    print(imgs.shape)
    print(idx)
    plt.imshow(imgs[0][0], cmap='gray', vmin=0, vmax=1)
    plt.show()
