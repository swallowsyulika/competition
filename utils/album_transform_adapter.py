import numpy as np

class TransformAdapter:
    """
    Adapt a albumentation transform so that it fits datasets designed with torchvision.transforms in mind.
    """

    def __init__(self, album_transform, input_type = 'PIL') -> None:
        self.album_transform = album_transform
        
        if input_type == 'PIL':
            self.preprocess = self._pil
        elif input_type == 'tensor':
            self.preprocess = self._tensor
    
    def _pil(self, x):
        # convert PIL image to numpy array
        return np.array(x)
    
    def _tensor(self, x):
        # convert Pytorch tensor to numpy array
        return x.permute(1, 2, 0).numpy()
        
    def __call__(self, x):

        x = self.preprocess(x)
        return self.album_transform(image=x)['image']
