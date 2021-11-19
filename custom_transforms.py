import random
import torch

from torchvision import transforms
import torchvision.transforms.functional as TF

from datasets.background_gen import BackgroundGenerator

class RandomLine:

    def __init__(self, size, amount, thickness = 3, num_samples=100, color = 0) -> None:
        
        self.size = size
        y_begin = (size - thickness) // 2
        self.amount = amount
        self.color = color
        
        self.line_tensor = torch.ones((1, size, size))
        self.line_tensor[0, y_begin:(y_begin + thickness), :] = 0

        self.transform = transforms.RandomAffine(
            degrees=(-180, 180),
            translate=(0.5, 0.5),
            shear=None,
            fill=1.0
            )

        # create mask samples
        print("caching random line...")
        self.masks = []

        for _ in range(num_samples):
            mask = torch.ones((1, size, size))
            num = random.randint(*self.amount)
            
            for _ in range(num):
                 mask*= self.transform(self.line_tensor)
            
            mask = ~mask.bool()
            self.masks.append(mask)

        
        
    def __call__(self, img):
        
        mask = random.choice(self.masks)        

        img[mask] = self.color

        return img

class RandomRatioPad:
    def __init__(self) -> None:
        pass

    def __call__(self, img):
        
        return TF.pad(img, (50, 0), fill=0)

class RandomDropResolution:
    def __init__(self, resolution_range) -> None:
        self.resolution_range = resolution_range        

    def __call__(self, img):
        res = random.randint(*self.resolution_range)
        img = TF.resize(img, res)
        return img
        
class RandomStackBackground:
    def __init__(self, bg_generator: BackgroundGenerator, ) -> None:
        self.bg_generator = bg_generator
    def __call__(self, img):

        size = img.shape[-1]
        texture = TF.to_tensor(self.bg_generator.get_PIL(size))

        # use softlight algorithm
        return 1 - ((1 - img) * (1 - texture))

        

