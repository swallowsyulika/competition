import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = True
from tqdm import tqdm

import config
from config import device
from utils import ensure_dir, denormalize
# paths
weight_dir = config.recognition_weights_path
log_dir = config.recognition_logs_path
use_cache = True

parser = argparse.ArgumentParser()
parser.add_argument('--weight', type=str, default=None, help='Specify the name of the pre-trained checkpoint to finetune.')
parser.add_argument('--num_epochs', type=int, default=10, help='Specify the number of epochs the model should be trained.')


args = parser.parse_args()



num_epochs = args.num_epochs
print(f"[I] Number of epochs: {num_epochs}")

if use_cache:
    from .preprocessed_dataset import dataset, characters
else:
    from .train_dataset import dataset, characters

ensure_dir(weight_dir)
ensure_dir(log_dir)
sample_weights = dataset.get_sample_weights()
for i in range(20, 5000, 1000):
    _, chid = dataset[i]
    print(f"{characters[chid]} {1/ dataset.class_weights[characters[chid]]} => {sample_weights[i]}")
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
print(len(sample_weights))

loader = DataLoader(dataset, batch_size=400, num_workers=4, pin_memory=True, sampler=sampler)
from .model import Net
net = Net(num_classes=len(characters)).to(device)

scalar = torch.cuda.amp.GradScaler()

def save_weight(name: str):
    checkpoint = {
        "net": net.state_dict(),
        "scaler": scalar.state_dict(),
    }
    torch.save(checkpoint, os.path.join(weight_dir, f"checkpoint_{name}.weight"))

def load_weight(name: str):
    weight_path = os.path.join(weight_dir, f"checkpoint_{name}.weight")
    checkpoint = torch.load(weight_path)
    net.load_state_dict(checkpoint["net"])
    scalar.load_state_dict(checkpoint["scaler"])
    print(f"[I] Successful loaded weight: {weight_path}")

if args.weight is not None:
    load_weight(args.weight)
else:
    print(f"[I] Weight file is not specified. Training from scratch.")

optimizer = optim.AdamW(net.parameters())
criterion = nn.CrossEntropyLoss()
writer = SummaryWriter(log_dir)
n_iters = 108510
try:
    for epoch in range(5, num_epochs):
        for character_imgs, classes in tqdm(loader):
            character_imgs_gpu = character_imgs.to(device)
            classes_gpu = classes.to(device)

            with torch.cuda.amp.autocast():
                optimizer.zero_grad()
                predictions = net(character_imgs_gpu) 
                loss = criterion(predictions, classes_gpu)
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
            # loss.backward()
            # optimizer.step()
            pred_ids = torch.argmax(predictions, dim=1).cpu()
            num_samples = character_imgs.size(0)
            num_correct = sum(pred_ids == classes)
            n_iters += 1
            writer.add_scalar("loss", loss.item(), n_iters)
            writer.add_scalar("accuracy", (num_correct / num_samples).item(), n_iters)
            writer.add_image("input", make_grid(denormalize(character_imgs[:16])), n_iters)
        save_weight(f"e_{epoch}")
except KeyboardInterrupt:
    save_weight(f"interrupt")

save_weight("final")