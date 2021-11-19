import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dcgan import Generator, Discriminator

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def gradient_penalty(critic, real, fake, dirty_imgs, device):
    batch_size, c, h, w = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    
    mixed_scores = critic(dirty_imgs, interpolated_images)
    gradient = torch.autograd.grad(
        inputs = interpolated_images,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim = 1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return gradient_penalty
    
def train():

    from train_dataset import dataset

    print(f"Number of samples: {len(dataset)}")

    weight_dir = "weights"
    # training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(dataset, batch_size=128, shuffle=True,
                        pin_memory=True, num_workers=2)
    
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    def save_weight(name: str):
        torch.save(netG.state_dict(), os.path.join(weight_dir, f"g_{name}.weight"))
        torch.save(netD.state_dict(), os.path.join(weight_dir, f"d_{name}.weight"))

    num_epochs = 101

    criterion = nn.BCEWithLogitsLoss()
    
    learning_rate = 1e-4
    lambda_gp = 10
    

    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.0, 0.9))

    n_critics = 1
    writer = SummaryWriter()
    n_iters = 0

    white_sum = 128 * 128
    
    try:
        for epoch in range(num_epochs):
            print(f"epoch: {epoch}")

            for dirty_imgs, clean_imgs in tqdm(loader):

                dirty_imgs = dirty_imgs.to(device)
                clean_imgs = clean_imgs.to(device)

                for _ in range(n_critics):
                    # train discriminator
                    netD.zero_grad()
                    # real
                    output = netD(dirty_imgs, clean_imgs)
                    # errD_real = criterion(output, torch.ones_like(output))                    
                    errD_real = -torch.mean(output)

                    # fake
                    fake_imgs = netG(dirty_imgs)
                    output = netD(dirty_imgs, fake_imgs.detach())
                    errD_fake = torch.mean(output)
                    
                    gp = gradient_penalty(netD, clean_imgs, output, dirty_imgs, device=device)
                    errD = errD_real + errD_fake + lambda_gp * gp
                    errD.backward(retain_graph=True)
                    optimizerD.step()

                # Parameter(Weight) Clipping for K-Lipshitz constraint
                writer.add_scalar("Loss D", errD.item(), n_iters)

                # train generator
                netG.zero_grad()
                generated_imgs = netG(dirty_imgs)
                output = netD(dirty_imgs, generated_imgs)
                errG = -torch.mean(output)
                errG.backward()
                optimizerG.step()

                writer.add_scalar("Loss G", errG.item(), n_iters)
                writer.add_image("G(img)", (generated_imgs[0][0] + 1) / 2.0, n_iters, dataformats='HW')
                writer.add_image("ground", (clean_imgs[0][0] + 1) / 2.0, n_iters, dataformats='HW')
                writer.add_image("img", (dirty_imgs[0][0] + 1) / 2.0, n_iters, dataformats='HW')

                n_iters += 1
            
            if epoch % 1 == 0:
                save_weight(f"e_{epoch}")            
        
        save_weight("final")



    except KeyboardInterrupt:
        print("Saving model...")
        save_weight("interrupt")


if __name__ == "__main__":
    train()