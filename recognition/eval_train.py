from torchvision import transforms
from model import Net
from eval_dataset import EvalDataset
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import file_to_list, parse_filename
# from train_dataset import characters

characters = sorted(file_to_list("characters_4808.txt") + file_to_list("train_extend.txt"))

device = "cuda" if torch.cuda.is_available() else "cpu"

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']

dataset = EvalDataset(
    characters,
    "/home/tingyu/projects/competition/gan/generated/output_residual",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((100, 100)),
        transforms.Normalize(std=0.5, mean=0.5)
    ]))
loader = DataLoader(dataset, batch_size=1500, shuffle=True,
                    num_workers=4, pin_memory=True)


def draw_preview(imgs, characters, predicted_classes, ground_classes, ncols=4):
    num_imgs = imgs.size(0)
    nrows = num_imgs // ncols
    if num_imgs % ncols > 0:
        nrows += 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    for i in range(num_imgs):
        img = imgs[i]

        predicted_label = characters[torch.argmax(predicted_classes[i])]
        ground_label = characters[ground_classes[i]]

        r = i // ncols
        c = i % ncols

        ax = axes[r, c]

        ax.imshow(img[0], cmap='gray', vmin=-1, vmax=1)
        if predicted_label == ground_label:
            ax.set_title(predicted_label)
        else:
            ax.set_title(f"{predicted_label}/{ground_label}")
    fig.tight_layout()
    return fig


weight_dir = "weights"
net = Net(num_classes=len(characters)).to(device)
net.load_state_dict(torch.load(os.path.join(weight_dir, "model_final.weight")))
net.eval()
criterion = nn.CrossEntropyLoss()
writer = SummaryWriter()
n_iters = 0
predicted = {}

with torch.no_grad():
    for character_imgs, classes in tqdm(loader):
        character_imgs_gpu = character_imgs.to(device)
        classes_gpu = classes.to(device)

        predictions = net(character_imgs_gpu)
        loss = criterion(predictions, classes_gpu)
        num_draw = 12
        fig = draw_preview(
            character_imgs[:num_draw], characters, predictions, classes)
        fig.savefig(f"generated/{n_iters}.jpg")
        n_iters += 1
        writer.add_scalar("loss", loss.item(), n_iters)
        writer.add_image("input", make_grid(character_imgs[:16]), n_iters)
