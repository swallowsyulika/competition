from typing import *
import os
import random

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath

from .qpainter_dataset_base import QPainterAbstractFontDataset

class FontLookupDatasetQt(QPainterAbstractFontDataset):
    def __init__(self,
                 characters: Sequence[str],
                 lookup_table: Sequence[str],
                 font_path: str,                 
                 font_size: int = 100,
                 img_size: int = 100,                 
                 random_character_color: bool = False,
                 transform=None):

        super().__init__(characters, font_path, img_size, font_size, None, transform)
        self.font_name = os.path.basename(font_path)
        self.lookup_table = lookup_table
        self.random_character_color = random_character_color

    def draw(self, painter: QPainter, path: QPainterPath):

        # generate random colors
        if self.random_character_color:
            text_color = random.randint(0, 255)
        else:
            text_color = 0

        # fill the path
        brush = QBrush()
        brush.setColor(QColor.fromRgb(text_color, text_color, text_color))
        brush.setStyle(Qt.SolidPattern)
        painter.fillPath(path, brush)

    def __getitem__(self, idx):
        img = super().__getitem__(idx)
        ch = self.characters[idx]
        return img, self.lookup_table[ch]