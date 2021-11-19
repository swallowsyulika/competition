import cv2

import os
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import Net
from utils import ensure_dir, denormalize
import config
from config import device
from .preprocessed_dataset import dataset, characters

cudnn.benchmark = True

loader = DataLoader(dataset, batch_size=1000, shuffle=False)
# from train_dataset import characters


# paths
weight_dir = config.recognition_weights_path
save_dir = config.recognition_err_check_path
ensure_dir(save_dir)


net = Net(num_classes=len(characters)).to(device)
checkpoint = torch.load(os.path.join(weight_dir, "checkpoint_e_2.weight"))
net.load_state_dict(checkpoint['net'])
net.eval()

with torch.no_grad():
    for idx, (character_imgs, character_ids) in tqdm(enumerate(loader)):

        b_size = character_imgs.size(0)

        # forward
        character_imgs_gpu = character_imgs.to(device)
        predictions = net(character_imgs_gpu)


        for i in range(b_size):
            ground_ch = characters[character_ids[i]] 
            pred_ch = characters[torch.argmax(predictions[i])]

            cv2.imwrite(os.path.join(save_dir, f"{ground_ch}=>{pred_ch}.png"), (denormalize(character_imgs[i][0]) * 255).numpy())