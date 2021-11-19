import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import cleaner_weights_path
from dcgan import Generator

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


from train_dataset import dataset

print(f"Number of samples: {len(dataset)}")

weight_dir = cleaner_weights_path
# training
device = "cuda" if torch.cuda.is_available() else "cpu"
loader = DataLoader(dataset, batch_size=400, shuffle=True,
                    pin_memory=True, num_workers=4)

netG = Generator().to(device)
netG.apply(weights_init)


def save_weight(name: str):
    torch.save(netG.state_dict(), os.path.join(weight_dir, f"g_{name}.weight"))

num_epochs = 50

criterion = nn.MSELoss()

optimizerG = optim.Adam(netG.parameters())

writer = SummaryWriter()
n_iters = 0

try:
    for epoch in range(num_epochs):
        print(f"epoch: {epoch}")

        for dirty_imgs, clean_imgs in loader:

            dirty_imgs = dirty_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            netG.zero_grad()
            output = netG(dirty_imgs)
            errG = criterion(output, clean_imgs)
            errG.backward()
            optimizerG.step()

            writer.add_scalar("MSE", errG.item(), n_iters)
            writer.add_image("G(img)", (output[0][0] + 1) / 2.0, n_iters, dataformats='HW')
            writer.add_image("img", (clean_imgs[0][0] + 1) / 2.0, n_iters, dataformats='HW')
            writer.add_image("dirty", (dirty_imgs[0][0] + 1) / 2.0, n_iters, dataformats='HW')

            n_iters += 1         
    
    save_weight("final")



except KeyboardInterrupt:
    print("Saving model...")
    save_weight("interrupt")
