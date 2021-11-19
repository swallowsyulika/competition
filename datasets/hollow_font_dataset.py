import os
import random

from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen, QPainter, QPainterPath
import numpy as np
from PIL import Image

from .background_gen import BackgroundGenerator
from .qpainter_dataset_base import QPainterAbstractFontDataset


class HollowFontDataset(QPainterAbstractFontDataset):
    
    def __init__(self,
                 characters,
                 font_path: str,
                 font_size:int = 100,
                 img_size: int =100,
                 border_size=(5, 10),
                 bg_generator: BackgroundGenerator = None,
                 transform=None):

        super().__init__(characters, font_path, img_size, font_size, bg_generator, transform)

        self.font_name = os.path.basename(font_path)
        self.border_size = border_size

    def draw(self, painter: QPainter, path: QPainterPath):

        # generate random colors
        border_color = random.randint(0, 255)

        # stroke the path
        pen = QPen()
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setColor(QColor.fromRgb(border_color, border_color, border_color))
        pen.setWidth(random.randint(*self.border_size))
        pen.setStyle(Qt.SolidLine)
        painter.strokePath(path, pen)