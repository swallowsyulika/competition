import csv
from utils.ensure_dir import ensure_dir

from torchvision import transforms
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import ImageDataset
from .model import Net
from utils import get_character_list, parse_filename, ensure_file
import config
from config import device
# from train_dataset import characters

ensure_file(config.eval_recognition_csv_path)

# paths
weight_dir = config.recognition_weights_path
preview_dir = config.eval_recognition_preview_path
ensure_dir(preview_dir)
save_preview = True
characters = sorted(get_character_list("characters_4808.txt") + get_character_list("train_extend.txt"))


plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']

def right_half(img):
    c, h, w = img.shape
    return img[:, :, w // 2:]
img_size = 64
dataset = ImageDataset(
    config.eval_cleaned_path,
    transform=transforms.Compose([
        transforms.ToTensor(),
        right_half,
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(std=0.5, mean=0.5)
    ]))
loader = DataLoader(dataset, batch_size=1500, shuffle=False,
                    num_workers=4, pin_memory=True)


def draw_preview(imgs, characters, predicted_classes, ncols=4):
    num_imgs = imgs.size(0)
    nrows = num_imgs // ncols
    if num_imgs % ncols > 0:
        nrows += 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    for i in range(num_imgs):
        img = imgs[i]

        predicted_label = characters[torch.argmax(predicted_classes[i])]

        r = i // ncols
        c = i % ncols

        ax = axes[r, c]

        ax.imshow(img[0], cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f"{predicted_label}")
    fig.tight_layout()
    return fig


net = Net(num_classes=len(characters)).to(device)
checkpoint = torch.load(os.path.join(weight_dir, "checkpoint_e_3.weight"))
net.load_state_dict(checkpoint['net'])
net.eval()
criterion = nn.CrossEntropyLoss()
writer = SummaryWriter()
predicted = {}

with torch.no_grad():
    for idx, (character_imgs, filenames) in tqdm(enumerate(loader)):

        b_size = character_imgs.size(0)

        # plt.imshow(character_imgs[0][0])
        # plt.show()

        # forward
        character_imgs_gpu = character_imgs.to(device)
        predictions = net(character_imgs_gpu)

        if save_preview:
            fig = draw_preview(character_imgs[:12], characters, predictions)
            fig.savefig(os.path.join(preview_dir, f"{idx}.jpg"))

        for i in range(b_size):
            img_name, container_id, orientation, character_id = parse_filename(filenames[i])

            # convert to int for easy sorting later
            container_id = int(container_id)
            character_id = int(character_id)

            predicted_idx = torch.argmax(predictions[i])
            ch = characters[predicted_idx]

            if img_name not in predicted:
                predicted[img_name] = {}
            
            if container_id not in predicted[img_name]:
                predicted[img_name][container_id] = {}
            
            predicted[img_name][container_id][character_id] = ch
# curate predicted
curated = {}
for img_name in predicted.keys():
    curated[img_name] = {}
    for container_id in sorted(predicted[img_name].keys()):
        container_text = ""
        for character_id in sorted(predicted[img_name][container_id]):
            container_text += predicted[img_name][container_id][character_id]
        curated[img_name][container_id] = container_text
print(curated)

filename_count = {}

with open(config.eval_csv_path, 'r') as f:
    csv_content = f.read()

with open(config.eval_recognition_csv_path, 'w') as f:
    lines = csv_content.split('\n')
    for line in lines:

        if len(line.strip()) == 0:
            continue

        filename = line.split(',')[0].strip()

        if filename not in filename_count:
            filename_count[filename] = 0
        else:
            filename_count[filename] += 1
        
        idx = filename_count[filename]

        if filename in curated:
            if idx in curated[filename]:
                f.write(f"{line},{curated[filename][idx]}\n")
            else:
                f.write(f"{line},###\n")
        else:
            f.write(f"{line},###\n")
