import os
import json

from utils import filter_character_list_advanced, ensure_file
from .character_list import characters
import config

# paths
dataset_dir = config.fonts_lib
cache_path = config.fonts_available_glyph_cache
ensure_file(cache_path)

def filter(name: str):
    """
    Determine whether the filename is a font file
    """
    lower_name = name.lower()
    if lower_name.endswith(".ttf"):
        return True
    if lower_name.endswith(".otf"):
        return True

font_files = []

for path, _, filenames in os.walk(dataset_dir):
    for filename in filenames:
        if filter(filename):
            font_files.append(os.path.join(path, filename))
            break

cache = {}

for idx, font in enumerate(font_files):
    font_name = os.path.split(font)[1]
    try:
        filtered = filter_character_list_advanced(font, characters)
    except OSError:
        print(f"[E] Unable to process {font_name}!")
        continue
    if font_name in cache:
        print(f"[!] Key {font_name} already exists!")
    cache[os.path.relpath(font, dataset_dir)] = filtered
    print(idx, font_name, len(filtered))

with open(cache_path, 'w', encoding='utf-8') as f:
    json.dump(cache, f, ensure_ascii=False)