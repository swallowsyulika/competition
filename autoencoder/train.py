import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from network_2lg import AutoEncoder

def train():

    from train_dataset import dataset

    print(f"Number of samples: {len(dataset)}")


    # training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(dataset, batch_size=1000, shuffle=True,
                        pin_memory=True, num_workers=4)
    ae = AutoEncoder().to(device)
    #ae.load_state_dict(torch.load("good5.weight"))
    print("weight loaded")

    num_epochs = 100

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(ae.parameters())

    writer = SummaryWriter()
    n_iters = 0
    try:
        for epoch in range(num_epochs):
            print(f"epoch: {epoch}")

            for dirty_imgs, clean_imgs in loader:

                dirty_imgs = dirty_imgs.to(device)
                clean_imgs = clean_imgs.to(device)

                outs = ae(dirty_imgs)
                loss = criterion(outs, clean_imgs)

                ae.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss", loss.item(), n_iters)
                writer.add_image("AE(x)", outs[0][0], n_iters, dataformats='HW')
                writer.add_image(
                    "Ground", clean_imgs[0][0], n_iters, dataformats='HW')
                writer.add_image(
                    "Dirty", dirty_imgs[0][0], n_iters, dataformats='HW')

                n_iters += 1

        torch.save(ae.state_dict(), "model_final.weight")

    except KeyboardInterrupt:
        print("Saving model...")
        torch.save(ae.state_dict(), "model_interrupt.weight")


if __name__ == "__main__":
    train()