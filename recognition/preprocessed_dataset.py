import os

from PIL import Image
from cv2 import mean
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.transforms.transforms import RandomAffine
from tqdm import std

import config
from utils import TransformAdapter
from custom_transforms import RandomLine
from .train_dataset import lookup_table, characters

# paths
dataset_dir = config.recognition_dataset_cache

class PreprocessedRecognitionDataset(Dataset):

    def __init__(self, dataset_dir: str, transform = None) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.transform = transform
        self.data = []
        self.class_weights = {}

        for character_dir in os.listdir(self.dataset_dir):

            character_files = os.listdir(os.path.join(self.dataset_dir, character_dir))

            self.class_weights[character_dir] =  1 / len(character_files)

            for character_file in character_files:

                self.data.append((character_dir, character_file))

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        character_dir, character_file = self.data[index]

        img = Image.open(os.path.join(self.dataset_dir, character_dir, character_file))

        if self.transform is not None:

            img = self.transform(img)
        
        return img, lookup_table[character_dir]

    def get_sample_weights(self):

        sample_weights = []

        for character_dir, _ in self.data:
            sample_weights.append(self.class_weights[character_dir])
        
        return sample_weights



import albumentations as A
from albumentations.pytorch import ToTensorV2

out_size = 64
# post_process = transforms.Compose([TransformAdapter(A.Compose([
#     A.Perspective(scale=(0.05, 0.1), pad_val=255),
#     # A.Resize(out_size, out_size),
#     A.Affine(
#         rotate=(-15, 15),
#         # translate=(0.2, 0.2),
#         scale=(0.95, 1.05),
#         # shear=5,
#         cval=255
#     ),
#     A.Normalize(mean=0.5, std=0.5),
#     ToTensorV2(),
# ])),
# # RandomLine(out_size, (0, 6), thickness=3, color=1)
# ])

post_process = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=(-15, 15), shear=5, fill=1),
    transforms.Normalize(mean=0.5, std=0.5)
])
dataset = PreprocessedRecognitionDataset(dataset_dir, transform=post_process)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils import denormalize
    for i in range(0, 1000, 100):
        img, ch = dataset[i]
        plt.imshow(denormalize(img[0]), vmin=0, vmax=1, cmap='gray')
        plt.show()
