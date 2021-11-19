import os
import random
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from typing import *

from .background_gen import BackgroundGenerator

class FontDataset(Dataset):
    def __init__(self,
                 characters: Sequence[str],
                 font_path: str,                 
                 font_size: int = 100,
                 img_size: int = 100,                 
                 bg_generator: BackgroundGenerator = None,
                 random_character_color: bool = False,
                 transform=None):

        super().__init__()
        self.img_size = img_size        
        self.font_name = os.path.basename(font_path)
        self.font_path = font_path
        self.transform = transform
        self.characters = characters
        self.bg_generator = bg_generator
        self.random_character_color = random_character_color
        self.font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
        print(f"number of characters: {len(self.characters)}")

    def __len__(self):
        return len(self.characters)

    def _draw_character(self, draw: ImageDraw, ch: str, color = 'black', offset_x: int = 0, offset_y: int = 0):
        W = self.img_size
        H = self.img_size        

        offset_w, offset_h = self.font.getoffset(ch)
        w, h = draw.textsize(ch, font=self.font)
        pos = ((W-w-offset_w)/2 + offset_x, (H-h-offset_h)/2 + offset_y)

        # Draw
        draw.text(pos, ch, color, font=self.font)

    def __getitem__(self, idx):

        ch = self.characters[idx]

        if self.bg_generator is not None:
            img = self.bg_generator.get_PIL(self.img_size)
        else:
            img = Image.new("L", (self.img_size, self.img_size), "white")

        draw = ImageDraw.Draw(img)

        if self.random_character_color:
            color = random.randint(0, 255)
            self._draw_character(draw, ch, color)
        
        else:
            self._draw_character(draw, ch)

        if self.transform is not None:
            img = self.transform(img)

        return img
