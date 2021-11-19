import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.serialization import save
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from .residual_gan import Generator
from datasets import ImageDataset
import config
from config import device
from utils import ensure_dir, ensure_file

parser = ArgumentParser()
parser.add_argument("--set",  choices=['train', 'eval'], default='eval', type=str)
parser.add_argument("--checkpoint", type=str)
args = parser.parse_args()

if args.set == "train":
    print("Evaluating training set...")
    dataset_root = config.train_characters_path
    out_dir = config.train_cleaned_path
elif args.set == "eval":
    print("Evaluating eval set....")
    dataset_root = config.eval_yolo_characters_path
    out_dir = config.eval_cleaned_path

weight_dir = config.cleaner_weights_path
checkpoint_name = args.checkpoint

ensure_dir(out_dir)

dataset = ImageDataset(dataset_root, transform=transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5),
]))

print(f"Number of samples: {len(dataset)}")



# eval
loader = DataLoader(dataset, batch_size=120, shuffle=False,
                    pin_memory=True, num_workers=4)
generator = Generator().to(device)

def load_weight(name: str):
    checkpoint = torch.load(os.path.join(weight_dir, f"checkpoint_{name}.weight"))
    generator.load_state_dict(checkpoint["generator"])
    print(f"loaded {name}")

load_weight(checkpoint_name)

print("weight loaded")
generator.eval()

criterion = nn.MSELoss()


to_pil = transforms.ToPILImage()

for imgs, rel_paths in tqdm(loader):
    
    b_size, c, h, w = imgs.shape
    
    
    outs = generator(imgs.to(device))
    outs = outs.cpu()

    outs[outs < 0] = -1 

    for idx in range(b_size):
        img, out, rel_path = imgs[idx], outs[idx], rel_paths[idx]
        
        concat_img = torch.zeros(h, w * 2)
        concat_img[:, 0:w] = img
        concat_img[:, w:] = out

        concat_img = to_pil((concat_img + 1)/2.0)
        save_path = os.path.join(out_dir, rel_path)
        save_dir = os.path.dirname(save_path)

        if not os.path.exists(save_dir):
            os.makedirs(os.path.dirname(save_path))

        concat_img.save(save_path)