from typing import *
import os
from random import randint
from tqdm import tqdm
from cv2 import sort
import torch
from PIL import Image
from PIL.ImageFont import load
from torchvision import transforms
import config
from datasets import ImageDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from .model import Net
from config import device
from utils import get_character_list

characters = sorted(get_character_list("characters_4808.txt") + get_character_list("train_extend.txt"))
def right_half(img):
    c, h, w = img.shape
    return img[:, :, w // 2:]


weight_dir = config.recognition_weights_path

dataset_path = config.train_cleaned_path

dataset = ImageDataset(dataset_path, transform=transforms.Compose([
    transforms.ToTensor(),
    right_half,
    transforms.Resize((64, 64)),
    transforms.Normalize(std=0.5, mean=0.5)
    ]))

net = Net(num_classes=len(characters)).to(device)
checkpoint = torch.load(os.path.join(weight_dir, "checkpoint_e_7.weight"))
net.load_state_dict(checkpoint['net'])
print("weight loaded")
net.eval()

def collate_fn(data):

    result = []
    for img, ch_path in data:
        character = os.path.dirname(ch_path)
        character_id = characters.index(character)
        result.append((img, character_id))
    
    return default_collate(result)


loader = DataLoader(dataset, batch_size=30, shuffle=True, pin_memory=True, collate_fn=collate_fn)

correct_dict = {}
num_dict = {}

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']

for index, (imgs, character_ids) in tqdm(enumerate(loader)):
    imgs_gpu = imgs.to(device)
    b_size = imgs.size(0)
    ncols = 5
    preds = net(imgs_gpu)
    pred_character_ids = torch.argmax(preds, dim=1)

    for i in range(b_size):

        character = characters[character_ids[i].item()]
        pred_character = characters[pred_character_ids[i].item()]

        if character not in correct_dict:
            correct_dict[character] = 0
            num_dict[character] = 0
        
        num_dict[character] += 1

        if character == pred_character: 
            correct_dict[character] += 1

    # fig, axes = plt.subplots(nrows=b_size // ncols, ncols=ncols, figsize=(30, 30))

    # for i in range(b_size):
    #     row = i // ncols
    #     col = i % ncols
    #     ax = axes[row][col]
    #     ax.imshow(imgs[i][0], vmin=-1, vmax=1, cmap='gray')
    #     ax.set_title(f"{characters[character_ids[i].item()]} > {characters[pred_character_ids[i].item()]}")
    #     fig.tight_layout()
    
    # plt.show()
    # print(pred_character_ids)

class CharacterRecord:
    def __init__(self, character: str, num_samples: int, num_correct: int) -> None:

        self.character = character
        self.num_samples = num_samples
        self.num_correct = num_correct
        self.accuracy = num_correct / num_samples

character_records: List[CharacterRecord] = []

for character in num_dict.keys():
    character_records.append(
        CharacterRecord(
            character,
            num_dict[character],
            correct_dict[character]
        )
    )

character_records.sort(key=lambda r : r.accuracy)    

with open("result.txt", 'w') as f:

    for record in character_records:
        f.write(f"{record.character} {record.accuracy} {record.num_correct}/{record.num_samples}\n")

