import os
import random

from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from torch.utils.data import Dataset
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QBrush, QColor, QImage, QPen, QRawFont, QPixmap, QPainter
import numpy as np
from PIL import Image

from .background_gen import BackgroundGenerator


def gen_ints(num, range, delta):

    values = []

    while len(values) < num:
        candidate = random.randrange(*range)
        is_candidate_valid = True

        for val in values:

            if abs(candidate - val) < delta:
                is_candidate_valid = False
                break

        if is_candidate_valid:
            values.append(candidate)

    return values


class TextureFontDataset(Dataset):
    
    def __init__(self,
                 characters,
                 font_path: str,
                 font_size:int = 100,
                 img_size: int =100,
                 border_size=(5, 10),
                 bg_generator: BackgroundGenerator = None,
                 transform=None):

        super().__init__()

        self.img_size = img_size
        self.font_name = os.path.basename(font_path)
        self.font_path = font_path
        self.transform = transform
        self.characters = characters
        self.border_size = border_size
        self.bg_generator = bg_generator

        # load font
        self.app = QApplication.instance()

        if self.app is None:
            # if it does not exist then a QApplication is created
            self.app = QApplication([])
        else:
            print("QApplication already exists.")

        self.font = QRawFont(font_path, font_size, QtGui.QFont.PreferNoHinting)

        print(f"number of characters: {len(self.characters)}")

    def __len__(self):
        return len(self.characters)

    def __getitem__(self, idx):
        # generate random colors
        text_color, border_color = gen_ints(2, (0, 255), 12)

        # get the unicode character to be painted
        ch = self.characters[idx]

        # convert the character to a glyph id
        glyph_id = self.font.glyphIndexesForString(ch)[0]

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

        # get font path and dimension
        path = self.font.pathForGlyph(glyph_id)
        bounding_rect = self.font.boundingRect(glyph_id)

        # calculate offsets
        x, y, w, h = bounding_rect.x(), bounding_rect.y(
        ), bounding_rect.width(), bounding_rect.height()

        pad_x = (self.img_size - w) // 2
        pad_y = (self.img_size - h) // 2

        # paint the path
        painter.translate(-x + pad_x, -y + pad_y)

        # stroke the path
        pen = QPen()
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setColor(QColor.fromRgb(border_color, border_color, border_color))
        pen.setWidth(random.randint(*self.border_size))
        pen.setStyle(Qt.SolidLine)
        painter.strokePath(path, pen)

        # fill the path
        brush = QBrush()
        
        texture_pil = self.bg_generator.gen_texture(self.img_size)
        texture_buf = np.array(texture_pil).tobytes()        
        texture = QImage(texture_buf, self.img_size, self.img_size, QImage.Format_Grayscale8)
        brush.setTextureImage(texture)
        
        #brush.setStyle(Qt.SolidPattern)
        painter.fillPath(path, brush)

        # release the painter
        painter.end()

        # convert the QPixmap to QImage
        gray_img = pixmap.toImage().convertToFormat(QtGui.QImage.Format_Grayscale8)

        # convert the QImage to a numpy array
        gray_np = np.array(gray_img.constBits().asarray(self.img_size * self.img_size)).reshape(
            self.img_size, self.img_size)

        # convert numpy array to PIL image
        img = Image.fromarray(gray_np)

        if self.transform is not None:
            img = self.transform(img)

        return img
