import os
from PIL import Image
from torch.utils.data import Dataset
from typing import *


class EvalDataset(Dataset):

    def __init__(self, characters: Sequence[str], root_dir: str, transform=None, right_only=True) -> None:
        super().__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.right_only = right_only
        self.data = []

        for character in os.listdir(self.root_dir):
            character_path = os.path.join(self.root_dir, character)
            try:
                idx = characters.index(character)
            except ValueError:
                print(f"{character} skipped.")
                continue
            if os.path.isdir(character_path) and len(character) == 1:
                for character_file in os.listdir(character_path):
                    self.data.append(
                        [os.path.join(character_path, character_file), idx])
            else:
                print(
                    f"[W] Unexpected file: {os.path.join(self.root_dir, character)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, id = self.data[index]
        img = Image.open(path)

        w, h = img.size

        if self.right_only:
            img = img.crop((w // 2, 0, w, h))

        if self.transform != None:
            img = self.transform(img)

        return img, id
