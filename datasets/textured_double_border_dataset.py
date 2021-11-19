import os
import random

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QImage, QPen, QPainter, QPainterPath
import numpy as np

from utils import gen_ints

from .qpainter_dataset_base import QPainterAbstractFontDataset
from .background_gen import BackgroundGenerator

class TextureDBorderFontDataset(QPainterAbstractFontDataset):
    
    def __init__(self,
                 characters,
                 font_path: str,
                 font_size:int = 100,
                 img_size: int =100,
                 border_size=(5, 10),
                 out_offset=(2,5),
                 shadow_offset_x = (-5, 5),
                 shadow_offset_y = (-5, 5),
                 bg_generator: BackgroundGenerator = None,
                 transform=None):

        super().__init__(characters, font_path, img_size, font_size, bg_generator, transform)

        self.font_name = os.path.basename(font_path)
        self.border_size = border_size
        self.out_offset = out_offset
        self.shadow_offset_x = shadow_offset_x
        self.shadow_offset_y = shadow_offset_y

    def draw(self, painter: QPainter, path: QPainterPath):
    
        border_color, outer_border_color = gen_ints(2, (0, 255), 100)

        # calculate random values
        stroke_width = random.randint(*self.border_size)
        out_offset = random.randint(*self.out_offset)
        
        # draw shadow (pseudo 3D)
        # fill + stroke
        shadow_offset_x = random.randint(*self.shadow_offset_x)
        shadow_offset_y = random.randint(*self.shadow_offset_y)
        
        painter.translate(shadow_offset_x, shadow_offset_y)
        shadow_color = random.randint(0, 255)
        brush = QBrush()
        brush.setStyle(Qt.SolidPattern)
        brush.setColor(QColor.fromRgb(shadow_color, shadow_color, shadow_color))
        painter.fillPath(path, brush)

        pen = QPen()
        pen.setStyle(Qt.SolidLine)
        pen.setWidth(stroke_width + out_offset)
        painter.strokePath(path, pen)
        
        painter.translate(-shadow_offset_x, -shadow_offset_y)


        # stroke the path (inner & outer)
        
        # outer border
        
        pen = QPen()
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setColor(QColor.fromRgb(outer_border_color, outer_border_color, outer_border_color))
        
        pen.setWidth(stroke_width + out_offset)
        pen.setStyle(Qt.SolidLine)
        painter.strokePath(path, pen)

        # inner border
        pen.setColor(QColor.fromRgb(border_color, border_color, border_color))
        pen.setWidth(stroke_width)
        painter.strokePath(path, pen)

        # fill the path
        brush = QBrush()
        
        texture_pil = self.bg_generator.gen_texture(self.img_size)
        texture_buf = np.array(texture_pil).tobytes()        
        texture = QImage(texture_buf, self.img_size, self.img_size, QImage.Format_Grayscale8)
        brush.setTextureImage(texture)
        
        painter.fillPath(path, brush)