from typing import *
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication
from torch.utils.data import Dataset
from PyQt5.QtGui import QPainter, QPainterPath, QRawFont, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt
from .background_gen import BackgroundGenerator

class QPainterAbstractFontDataset(Dataset):

    def __init__(self, characters: Sequence[str], font_path: str, img_size: int, font_size: int, bg_generator: BackgroundGenerator , transform = None) -> None:
        super().__init__()
        self.characters = characters
        self.img_size = img_size
        self.font_size = font_size
        self.font_path = font_path
        self.ready = False
        self.bg_generator = bg_generator
        self.transform = transform
    
    def init(self):
        """
        We don't want to put these in __init__() because it would cause problems when
        the dataset is used along with num_workers > 0 on Windows platforms.
        Windows platform does not have fork(), making it much more troublesome to copy
        complex objects such as QRawFont, which is, unfortunately, required by QPainter
        based font dataset.
        """
        # load font
        if not QApplication.instance():
            raise ValueError("Please create an instance of QApplication on your main program.")


        self.font = QRawFont(self.font_path, self.font_size, QFont.PreferNoHinting)

        print(f"number of characters: {len(self.characters)}")
        self.ready = True
    
    def __len__(self):
        return len(self.characters)
    
    def draw(self, painter: QPainter, path: QPainterPath):
        pass

    def __getitem__(self, idx):

        if not self.ready:
            self.init()

        # create a QPixmap
        if self.bg_generator is not None:
            buffer = self.bg_generator.get_numpy(self.img_size).tobytes()
            qimg = QImage(buffer, self.img_size, self.img_size, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)
            
        else:
            pixmap = QPixmap(self.img_size, self.img_size)            
            pixmap.fill(Qt.white)

        # create a QPainter
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # get the unicode character to be painted
        ch = self.characters[idx]

        # convert the character to a glyph id
        glyph_id = self.font.glyphIndexesForString(ch)[0]

        # get font path and dimension
        path = self.font.pathForGlyph(glyph_id)
        bounding_rect = self.font.boundingRect(glyph_id)

        # calculate offsets
        x, y, w, h = bounding_rect.x(), bounding_rect.y(
        ), bounding_rect.width(), bounding_rect.height()

        pad_x = (self.img_size - w) // 2
        pad_y = (self.img_size - h) // 2

        painter.translate(-x + pad_x, -y + pad_y)

        self.draw(painter, path)

        # release the painter
        painter.end()

        # convert the QPixmap to QImage
        gray_img = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)

        # convert the QImage to a numpy array
        gray_np = np.array(gray_img.constBits().asarray(self.img_size * self.img_size)).reshape(
            self.img_size, self.img_size)

        # convert numpy array to PIL image
        img = Image.fromarray(gray_np)

        if self.transform is not None:
            img = self.transform(img)

        return img