import os
import json
from PIL import Image
from torch.utils import data
from datasets.font_dataset import FontDataset
import config
from utils import get_character_list, ensure_dir, filter_character_list_advanced
from datasets import FontLookupDataset

edu_4808 = get_character_list("characters_4808.txt")
extended = get_character_list("train_extend.txt")

characters = ["當", "電", "體"] + extended

black_listed = []

with open(config.font_lib_blacklist_path, 'r') as f:
    black_listed = f.read().split("\n")

print("blacklisted")     
print(black_listed)

print(characters)
ensure_dir(config.fonts_lib_manual_check_path)
by_characters_path = os.path.join(config.fonts_lib_manual_check_path, "by_ch")
by_fonts_path = os.path.join(config.fonts_lib_manual_check_path, "by_fonts")


with open(config.fonts_available_glyph_cache, 'r') as f:
    cache = json.load(f)

def is_in_blacklist(font_path: str):
    for name in black_listed:
        if name in font_path:
            return True
    return False

for font_path in cache.keys():
    if is_in_blacklist(font_path):
        print(f"[BlackList] Skipped {font_path}.")
        continue
    full_font_path = os.path.join(config.fonts_lib, font_path)
    cs = filter_character_list_advanced(full_font_path, characters)
    dataset = FontDataset(cs, full_font_path)

    for index, character in enumerate(cs):

        img = dataset[index]

        save_path = os.path.join(by_characters_path, character)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path2 = os.path.join(by_fonts_path, os.path.basename(font_path))
        if not os.path.exists(save_path2):
            os.makedirs(save_path2)

        img.save(os.path.join(save_path, os.path.basename(font_path) + ".png"))
        img.save(os.path.join(save_path2, character + ".png"))
