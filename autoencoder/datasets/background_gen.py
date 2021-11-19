import random
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from typing import *

class BackgroundGenerator:

    def __init__(self, texture_dir: str, zones_range: Tuple[int, int]=(2, 5)) -> None:
        self.texture_dir = texture_dir
        self.zones_range = zones_range
        self.texture_files = [x for x in os.listdir(texture_dir) if x.endswith(".jpg") or x.endswith(".png")]
        self.generators = [self.gen_texture, self.gen_solid, self.gen_horizontal_zones, self.gen_vertical_zones]
    
    
    def gen_texture(self, size: int):
        texture_file = random.choice(self.texture_files)
        img = Image.open(os.path.join(self.texture_dir, texture_file)).convert('L')
        img = img.resize((size, size))
        return img
    
    def gen_solid(self, size: int):
        color = random.randint(0, 255)
        img = Image.new('L', (size, size), color)
        return img

    
    def gen_rect_zones_data(self, size: int):
        num_zones = random.randint(*self.zones_range)

        if num_zones <= 1:
            return self.gen_solid()

        
        
        points = np.random.choice(size, num_zones - 1, False)
        points = [0] + points.tolist() + [size]
        
        backgrounds = np.random.choice(255, num_zones, False).tolist()

        

        zone_data = []

        
        for i in range(len(points) - 1):
            begin = points[i]
            end = points[i + 1]
            background = backgrounds[i]
            zone_data.append((begin, end, background))
        
        return zone_data
            
                
    def gen_horizontal_zones(self, size):
        img = Image.new('L', (size, size))
        draw = ImageDraw.Draw(img)
        zone_data = self.gen_rect_zones_data(size)
        for begin, end, background in zone_data:
            draw.rectangle([0, begin, size, end], fill=background)
        
        return img


    def gen_vertical_zones(self, size):
        img = Image.new('L', (size, size))
        draw = ImageDraw.Draw(img)
        zone_data = self.gen_rect_zones_data(size)
        for begin, end, background in zone_data:
            draw.rectangle([begin, 0, end, size], fill=background)
        
        return img


    def get_PIL(self, size: int):
        generator = random.choice(self.generators)
        return generator(size)

    def get_numpy(self, size):
        return np.array(self.get_PIL(size))



if __name__ == '__main__':
    generator = BackgroundGenerator("/home/tingyu/projects/competition/textures/", zones_range=(10, 30))

    fig, axes = plt.subplots(nrows=len(generator.generators), ncols=1)

    for i in range(len(axes)):
        img = np.array(generator.generators[i](128))
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
    
    plt.show()