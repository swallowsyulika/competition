import os
from config import project_character_lists_path
def get_character_list(name: str):
    with open(os.path.join(project_character_lists_path, name), 'r', encoding='utf-8') as f:
        characters_raw = f.read()
    return [characters_raw[x] for x in range(len(characters_raw))]