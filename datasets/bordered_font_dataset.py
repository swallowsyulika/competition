import os
import random

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPen, QPainter, QPainterPath

from utils import gen_ints

from .background_gen import BackgroundGenerator
from .qpainter_dataset_base import QPainterAbstractFontDataset



class BorderedFontDataset(QPainterAbstractFontDataset):
    
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
        text_color, border_color = gen_ints(2, (0, 255), 12)


        # stroke the path
        pen = QPen()
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setColor(QColor.fromRgb(border_color, border_color, border_color))
        pen.setWidth(random.randint(*self.border_size))
        pen.setStyle(Qt.SolidLine)
        painter.strokePath(path, pen)

        # fill the path
        brush = QBrush(QColor.fromRgb(text_color, text_color, text_color))
        brush.setStyle(Qt.SolidPattern)
        painter.fillPath(path, brush)
