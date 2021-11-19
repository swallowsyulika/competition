import os
import json
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.geometric.transforms import Perspective
import torchvision.transforms as transforms

from datasets import ConcatDataset, FontLookupDatasetQt
from config import fonts_lib, fonts_available_glyph_cache
from .character_list import characters

import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import TransformAdapter
import config

lookup_table = {characters[index]: index for index in range(len(characters))}

dataset_dir = fonts_lib

with open(fonts_available_glyph_cache, 'r', encoding='utf-8') as f:
    cache = json.load(f)

font_size = 128
out_size = 64
datasets = []

def get_post_process(rotate_deg: int):
    return TransformAdapter(A.Compose([
        # A.Perspective(scale=(0.05, 0.1), pad_val=255),
        A.Resize(out_size, out_size),
        A.Affine(
            rotate=rotate_deg,
            # translate=(0.2, 0.2),
            # scale=(0.95, 1.05),
            # shear=5,
            cval=255
        ),
        A.Resize(128, 128),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ]))

with open(config.font_lib_blacklist_path, 'r') as f:
    black_listed = f.read().split("\n")

print("blacklisted")     
print(black_listed)

def is_in_blacklist(font_path: str):
    for name in black_listed:
        if name in font_path:
            return True
    return False

for font_file in cache.keys():
    if is_in_blacklist(font_file):
        print(f"[BlackList] Skipped {font_file}.")
        continue
    filtered_characters = cache[font_file]
    for deg in (0,):
        try:
            ds = FontLookupDatasetQt(filtered_characters, lookup_table, os.path.join(dataset_dir, font_file),
                            font_size=font_size,
                            img_size=font_size,
                            random_character_color=False,
                            transform=get_post_process(deg))
        except:
            print(f"Unable to open file: {os.path.join(dataset_dir, font_file)}.")
            continue

        datasets.append(ds)

dataset = ConcatDataset(*datasets)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    clean, dirty = dataset[10]

    plt.imshow(clean[0])
    plt.show()

    plt.imshow(dirty[0])
    plt.show()
