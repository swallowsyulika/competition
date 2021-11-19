import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']

device = "cuda" if torch.cuda.is_available() else "cpu"

class CharacterDataset(Dataset):
    def __init__(self, root_dir: str, cache_dir: str, transform):
        super().__init__()
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.transform = transform
        self.characters = sorted([x for x in os.listdir(root_dir)])
        print(f"number of characters: {len(self.characters)}")
        self.data = []
        for idx, ch in enumerate(self.characters):
            for ch_file in os.listdir(os.path.join(root_dir, ch)):
                if os.path.isfile(os.path.join(root_dir, ch, ch_file)):
                    self.data.append((idx, ch_file))
        
        print(f"number of data: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        ch_idx, ch_file = self.data[idx]
        ch_path = os.path.join(self.root_dir, self.characters[ch_idx], ch_file)
        img = Image.open(ch_path).convert('L')
        
        if self.transform is not None:
            img = self.transform(img)

        #category = torch.zeros(len(self.characters))
        #category[ch_idx] = 1
        
        #return img, category
        
        return img, os.path.join(self.cache_dir, self.characters[ch_idx], ch_file)



class FontDataset(Dataset):
    def __init__(self, characters, font_path: str, cache_dir: str, font_size=100, img_size=100, transform = None):
        super().__init__()
        self.img_size = img_size
        self.font_name = os.path.basename(font_path)
        self.font_path = font_path        
        self.cache_dir = cache_dir
        self.transform = transform
        self.characters = characters
        self.font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
        print(f"number of characters: {len(self.characters)}")
    
    def __len__(self):
        return len(self.characters)


    def __getitem__(self, idx):
        ch = self.characters[idx]
        W = self.img_size
        H = self.img_size
        img = Image.new("L", (self.img_size, self.img_size), "white")
        draw = ImageDraw.Draw(img)
        offset_w, offset_h = self.font.getoffset(ch)
        w, h = draw.textsize(ch, font=self.font)
        pos = ((W-w-offset_w)/2, (H-h-offset_h)/2)
        # Draw
        draw.text(pos, ch, "black", font=self.font)

        if self.transform is not None:
            img = self.transform(img)

        
        return img, os.path.join(self.cache_dir, ch, f"{self.font_name}.png")

    

transform = transforms.Compose([
    transforms.ToTensor(),
    # FastRemoveBorder(300),
    transforms.ToPILImage(),
])

dataset = CharacterDataset("Handwritten_Data", "Preprocessed_Data", transform = transform)
font_dataset = FontDataset(dataset.characters, "/home/tingyuyan/Downloads/JasonHandWritingFonts-20210716/tw/JasonHandwriting3-ExtraLight.ttf", "_Preprocessed_Data")

num_classes = len(dataset.characters)

batch_size = 2400


def c(x):
    return x

trainloader = torch.utils.data.DataLoader(font_dataset, batch_size=batch_size,
                                          shuffle=False, collate_fn=c, num_workers=1)

for x in tqdm(trainloader):
    for img, path in x:
        tensor = transforms.ToTensor()(img)
        if (tensor == 0).sum() <= 0:
            print("[!] no character found in the img")
        else:
            if not os.path.exists(os.path.dirname(path)):
                try:
                    os.makedirs(os.path.dirname(path))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            with open(path, 'wb') as f:
                img.save(f)

# print(imgs[0])

# plt.imshow(imgs[0][0], cmap='gray', vmin = 0, vmax=1)
# plt.show()

#grid = make_grid(imgs)
#save_image(grid, "grid.png")