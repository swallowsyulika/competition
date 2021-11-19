import os
import random

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath

from .background_gen import BackgroundGenerator
from .qpainter_dataset_base import QPainterAbstractFontDataset

class FontDatasetQt(QPainterAbstractFontDataset):
    
    def __init__(self,
                 characters,
                 font_path: str,
                 font_size:int = 100,
                 img_size: int =100,
                 bg_generator: BackgroundGenerator = None,
                 random_character_color: bool = False,
                 transform=None):

        super().__init__(characters, font_path, img_size, font_size, bg_generator, transform)

        self.font_name = os.path.basename(font_path)
        self.random_character_color = random_character_color

    def draw(self, painter: QPainter, path: QPainterPath):
        # fill the path
        brush = QBrush()
        if self.random_character_color:
            c = random.randint(0, 255)
            brush.setColor(QColor.fromRgb(c, c, c))
        else:
            brush.setColor(Qt.black)
        brush.setStyle(Qt.SolidPattern)
        painter.fillPath(path, brush)


if __name__ == "__main__":
    import config

    ds = FontDatasetQt(['測', '試'], os.path.join(config.project_fonts_path, 'NotoSansCJK-Medium.ttc'))

    img = ds[0]

    img.save("sample.png")