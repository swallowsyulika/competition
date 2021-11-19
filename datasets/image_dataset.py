import os
from PIL import Image
from torch.utils.data import Dataset
class ImageDataset(Dataset):

    def __init__(self, root_dir: str, transform = None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.images = []
        self.transform = transform

        # find images recursively
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                self.images.append(os.path.join(root, f))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        rel_path = os.path.relpath(img_path, self.root_dir)

        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, rel_path


if __name__ == "__main__":
    ds = ImageDataset("/home/tingyu/generated/train/characters/")

    img, path = ds[10]
    print(path)
