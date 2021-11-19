from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from .train_dataset import dataset
from utils import denormalize


loader = DataLoader(dataset, shuffle=True, batch_size=25)

for dirty_imgs, cleaned_imgs in loader:
    grid = make_grid(denormalize(dirty_imgs))

    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.show()



