import os

import os
from torch.utils import data
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms

class CharacterImageDataset(Dataset):
    def __init__(self, root_dir, transform = None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        for character in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir,character)):
                for ch_file in os.listdir(os.path.join(self.root_dir, character)):
                    if os.path.isfile(os.path.join(self.root_dir, character, ch_file)):
                        self.data.append((character, ch_file))
        
        

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        ch, ch_file = self.data[index]

        img = Image.open(os.path.join(self.root_dir, ch, ch_file)).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img, ch
        