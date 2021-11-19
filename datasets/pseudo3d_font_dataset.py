import random
from PIL import Image, ImageDraw, ImageFont
from typing import *
from .background_gen import BackgroundGenerator
from .font_dataset import FontDataset

class Pseudo3DFontDataset(FontDataset):
    def __init__(self,
                 characters: Sequence[str],
                 font_path: str,                 
                 font_size: int = 100,
                 img_size: int = 100,                 
                 bg_generator: BackgroundGenerator = None,                 
                 x_offset_range = (-5, 5),
                 y_offset_range = (-5, 5),
                 transform=None):
        super().__init__(characters, font_path, font_size=font_size, img_size=img_size, bg_generator=bg_generator, random_character_color=False, transform=transform)
        self.x_offset_range = x_offset_range
        self.y_offset_range = y_offset_range
    
    def __getitem__(self, idx):

        ch = self.characters[idx]

        if self.bg_generator is not None:
            img = self.bg_generator.get_PIL(self.img_size)
        else:
            img = Image.new("L", (self.img_size, self.img_size), "white")

        draw = ImageDraw.Draw(img)

        x_offset = random.randint(*self.x_offset_range)
        y_offset = random.randint(*self.y_offset_range)

        
               
        text_color = random.randint(0, 255)
        shadow_color = random.randint(0, 255)
        
        self._draw_character(draw, ch, shadow_color, offset_x=x_offset, offset_y=y_offset)
        self._draw_character(draw, ch, text_color)
        

        if self.transform is not None:
            img = self.transform(img)

        return img

        