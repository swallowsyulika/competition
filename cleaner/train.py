import os
import argparse
from PyQt5.QtWidgets import QApplication
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from PyQt5.QtCore import Qt

app = QApplication([])

import config
from config import device
from utils import denormalize, ensure_dir

from .residual_gan import Generator, Discriminator
if __name__ == "__main__":
    from .train_dataset import dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default=None, help='Specify the name of the pre-trained checkpoint to finetune.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Specify the number of epochs the model should be trained.')


    args = parser.parse_args()
    print(f"Number of samples: {len(dataset)}")

    # paths
    weight_dir = config.cleaner_weights_path
    log_dir = config.cleaner_log_path

    ensure_dir(weight_dir)
    ensure_dir(log_dir)

    cudnn.benchmark = True

    # training
    loader = DataLoader(dataset, batch_size=128, shuffle=True,
                        pin_memory=True, num_workers=2)

    netG = Generator().to(device)
    netD = Discriminator().to(device)



    #netG.load_state_dict(torch.load("weights/g_e_3.weight"))
    #netD.load_state_dict(torch.load("weights/d_e_3.weight"))

    scalarG = torch.cuda.amp.GradScaler()
    scalarD = torch.cuda.amp.GradScaler()

    def save_weight(name: str):
        checkpoint = {
            "generator": netG.state_dict(),
            "discriminator": netD.state_dict(),
            "scalerG": scalarG.state_dict(),
            "scalarD": scalarD.state_dict(),
        }
        torch.save(checkpoint, os.path.join(weight_dir, f"checkpoint_{name}.weight"))

    def load_weight(name: str):
        weight_path = os.path.join(weight_dir, f"checkpoint_{name}.weight")
        checkpoint = torch.load(weight_path)
        netG.load_state_dict(checkpoint["generator"])
        netD.load_state_dict(checkpoint["discriminator"])
        scalarG.load_state_dict(checkpoint["scalerG"])
        scalarD.load_state_dict(checkpoint["scalarD"])
        print(f"[I] Successful loaded weight: {weight_path}")

    if args.weight is not None:
        load_weight(args.weight)
    else:
        print(f"[I] Weight file is not specified. Training from scratch.")


    num_epochs = args.num_epochs
    print(f"[I] Number of epochs: {num_epochs}")

    criterion = nn.MSELoss()
        
    learning_rate = 1e-5


    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    n_critics = 1
    n_generator = 2
    writer = SummaryWriter(log_dir)
    n_iters = 0

    white_sum = 128 * 128

    tensorboard_samples = 8


    try:
        for epoch in range(1, num_epochs):
            print(f"epoch: {epoch}")

            for dirty_imgs, clean_imgs in tqdm(loader):

                dirty_imgs = dirty_imgs.to(device)
                clean_imgs = clean_imgs.to(device)

                for _ in range(n_critics):
                    with torch.cuda.amp.autocast():
                        # train discriminator
                        netD.zero_grad()
                        # real
                        output = netD(dirty_imgs, clean_imgs)
                        errD_real = criterion(output, torch.ones_like(output))                        

                        # fake
                        fake_imgs = netG(dirty_imgs)
                        
                        output = netD(dirty_imgs, fake_imgs.detach())
                        errD_fake = criterion(output, torch.zeros_like(output))
                                        
                    errD = errD_real + errD_fake
                    scalarD.scale(errD).backward()
                    scalarD.step(optimizerD)
                    scalarD.update()
                    #errD.backward()
                    #optimizerD.step()

                # Parameter(Weight) Clipping for K-Lipshitz constraint
                writer.add_scalar("Loss D", errD.item(), n_iters)

                # train generator              
                for _ in range(n_generator):  
                    with torch.cuda.amp.autocast():
                        netG.zero_grad()
                        generated_imgs = netG(dirty_imgs)
                        output = netD(dirty_imgs, generated_imgs)
                        errG_disc = criterion(output, torch.ones_like(output))
                        errG_reconstruct = criterion(generated_imgs, clean_imgs)

                        errG = errG_disc + errG_reconstruct
                    scalarG.scale(errG).backward()
                    scalarG.step(optimizerG)
                    scalarG.update()
                    #errG.backward()
                    #optimizerG.step()
                
                if n_iters % 2 == 0:
                    writer.add_scalar("Loss G", errG.item(), n_iters)
                    writer.add_scalar("Loss G (disc)", errG_disc.item(), n_iters)
                    writer.add_scalar("Loss G (reconstruct)", errG_reconstruct.item(), n_iters)
                    
                    generated_imgs = denormalize(generated_imgs[:tensorboard_samples])
                    ground_imgs = denormalize(clean_imgs[:tensorboard_samples])
                    input_imgs = denormalize(dirty_imgs[:tensorboard_samples])

                    combined = make_grid(torch.cat((input_imgs, generated_imgs, ground_imgs), dim=0))
                    writer.add_image("preview", combined, n_iters, dataformats='CHW')


                n_iters += 1
            
            if epoch % 1 == 0:
                save_weight(f"e_{epoch}")            
        
        save_weight("final")

    except KeyboardInterrupt:
        print("Saving model...")
        save_weight("interrupt")
