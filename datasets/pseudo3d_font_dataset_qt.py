from typing import *
import os
import random

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath

from .background_gen import BackgroundGenerator
from .qpainter_dataset_base import QPainterAbstractFontDataset

class Pseudo3DFontDatasetQt(QPainterAbstractFontDataset):
    
    def __init__(self,
                 characters: Sequence[str],
                 font_path: str,                 
                 font_size: int = 100,
                 img_size: int = 100,                 
                 bg_generator: BackgroundGenerator = None,                 
                 x_offset_range = (-5, 5),
                 y_offset_range = (-5, 5),
                 transform=None):

        super().__init__(characters, font_path, img_size, font_size, bg_generator, transform)

        self.font_name = os.path.basename(font_path)
        self.x_offset_range = x_offset_range
        self.y_offset_range = y_offset_range

    def draw(self, painter: QPainter, path: QPainterPath):
        # random colors and offsets
        text_color = random.randint(0, 255)
        shadow_color = random.randint(0, 255)

        x_offset = random.randint(*self.x_offset_range)
        y_offset = random.randint(*self.y_offset_range)

        # fill the path
        painter.translate(x_offset, y_offset)
        brush = QBrush()
        brush.setStyle(Qt.SolidPattern)
        brush.setColor(QColor.fromRgb(shadow_color, shadow_color, shadow_color))
        painter.fillPath(path, brush)

        painter.translate(-x_offset, -y_offset)
        brush.setColor(QColor.fromRgb(text_color, text_color, text_color))
        painter.fillPath(path, brush)


if __name__ == "__main__":
    import config

    ds = Pseudo3DFontDatasetQt(['測', '試'], os.path.join(config.project_fonts_path, 'NotoSansCJK-Medium.ttc'))

    img = ds[0]

    img.save("sample.png")