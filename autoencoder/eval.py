import os
import torch
from torch import random
import torch.nn as nn
from torch.serialization import save

from torchvision import transforms
from torch.utils.data import DataLoader

from network_2lg import AutoEncoder
from eval_dataset import dataset

print(f"Number of samples: {len(dataset)}")

out_dir = "generated/output"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# eval
device = "cuda" if torch.cuda.is_available() else "cpu"
loader = DataLoader(dataset, batch_size=250, shuffle=True,
                    pin_memory=True, num_workers=4)
ae = AutoEncoder().to(device)
ae.load_state_dict(torch.load("model_interrupt.weight"))
print("weight loaded")
ae.eval()

criterion = nn.MSELoss()


to_pil = transforms.ToPILImage()

for imgs, chs in loader:
    
    b_size, c, h, w = imgs.shape
    
    
    outs = ae(imgs.to(device))
    outs = outs.cpu()

    for idx in range(b_size):
        img, out, ch = imgs[idx], outs[idx], chs[idx]
        
        concat_img = torch.zeros(h, w * 2)
        concat_img[:, 0:w] = img
        concat_img[:, w:] = out

        save_path = os.path.join(out_dir, ch)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        num_samples = len(os.listdir(save_path))

        concat_img = to_pil(concat_img)
        concat_img.save(os.path.join(save_path, f"{num_samples + 1}.png"))


