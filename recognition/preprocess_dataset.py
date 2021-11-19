from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import config
import cv2
from .train_dataset import dataset, lookup_table
from utils import ensure_dir


save_dir = config.recognition_dataset_cache
ensure_dir(save_dir)

lookup_table_reverse = {key: value for (value, key) in lookup_table.items() }

def collate_fn(data):
    for img, character_id in data:
        character = lookup_table_reverse[character_id]
        save_path = os.path.join(save_dir, character)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        num_files = len(os.listdir(save_path))
        cv2.imwrite(os.path.join(save_path, f"{num_files}.jpg"), img)
        

loader = DataLoader(dataset, batch_size=30, num_workers=15, shuffle=False, collate_fn=collate_fn)


for _ in tqdm(loader):
    pass
