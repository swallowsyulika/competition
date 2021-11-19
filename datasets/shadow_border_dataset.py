import os
import random

from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from matplotlib import use
from torch.utils.data import Dataset
from PyQt5.QtWidgets import QApplication, QGraphicsDropShadowEffect, QGraphicsScene, QGraphicsView
from PyQt5.QtGui import QBrush, QColor, QImage, QPen, QRawFont, QPixmap, QPainter, QFont
import numpy as np
from PIL import Image

from .background_gen import BackgroundGenerator

class ShadowBorderDataset(Dataset):
    
    def __init__(self,
                 characters,
                 font_path: str,
                 font_size:int = 100,
                 img_size: int =100,
                 border_size=(10, 20),
                 shadow_offset_x = (-5, 5),
                 shadow_offset_y = (-5, 5),
                 shadow_radius = (10, 15),
                 shadow_color = (0, 100),
                 bg_generator: BackgroundGenerator = None,
                 use_textured_text: bool = True,
                 transform=None):

        super().__init__()

        self.img_size = img_size
        self.font_size = font_size
        self.font_name = os.path.basename(font_path)
        self.font_path = font_path
        self.transform = transform
        self.characters = characters
        self.border_size = border_size
        self.bg_generator = bg_generator
        self.shadow_offset_x = shadow_offset_x
        self.shadow_offset_y = shadow_offset_y
        self.shadow_radius = shadow_radius
        self.shadow_color = shadow_color
        self.use_textured_text = use_textured_text

        # self.graphics_view = QGraphicsView()

        self.ready = False

    
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
        self.graphics_scene = QGraphicsScene()

        print(f"number of characters: {len(self.characters)}")
        self.ready = True

    def __len__(self):
        return len(self.characters)

    def __getitem__(self, idx):
        if not self.ready:
            self.init()
            self.ready = True
        # generate random colors
        border_color = random.randint(200, 255)

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


        # # stroke the path (inner & outer)
        self.graphics_scene.clear()

        border_pen = QPen()
        border_pen.setColor(QColor.fromRgb(border_color, border_color, border_color))
        border_pen.setWidth(random.randint(*self.border_size))

        effect = QGraphicsDropShadowEffect()
        effect.setBlurRadius(random.randint(*self.shadow_radius))
        shadow_color = random.randint(*self.shadow_color)
        effect.setColor(QColor.fromRgb(shadow_color, shadow_color, shadow_color))
        effect.setXOffset(random.randint(*self.shadow_offset_x))
        effect.setYOffset(random.randint(*self.shadow_offset_y))

        empty_pen = QPen()
        empty_pen.setWidth(0)
        empty_pen.setColor(QColor.fromRgb(0, 0, 0, 0))

        # draw text border along with drop shadow
        self.graphics_scene.addRect(x - pad_x, y - pad_y, self.img_size, self.img_size, pen=empty_pen)
        border_item = self.graphics_scene.addPath(path, pen=border_pen)
        border_item.setGraphicsEffect(effect)
        # draw the text itself
        brush = QBrush()
        brush.setStyle(Qt.SolidPattern) 

        if self.use_textured_text:
            texture_pil = self.bg_generator.gen_texture(self.img_size)
            texture_buf = np.array(texture_pil).tobytes()        
            texture = QImage(texture_buf, self.img_size, self.img_size, QImage.Format_Grayscale8)
            brush.setTextureImage(texture)
        else:
            brush.setColor(Qt.black)

        self.graphics_scene.addPath(path, brush=brush, pen=empty_pen)

        # self.graphics_view.setScene(self.graphics_scene)
        # self.graphics_view.show()
        
        # release the painter
        self.graphics_scene.setSceneRect(self.graphics_scene.itemsBoundingRect())
        self.graphics_scene.render(painter)
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

if __name__ == "__main__":
    import config
    from .background_gen import BackgroundGenerator
    app = QApplication([])
    bg_generator = BackgroundGenerator(config.project_textures_path)
    ds = ShadowBorderDataset(
        ['測', '試', '卑'], 
        os.path.join(config.project_fonts_path, "TaipeiSansTCBeta-Bold.ttf"),
        font_size=96,
        img_size=128,
        bg_generator = bg_generator
        )

    for i in range(len(ds)):
        img = ds[i]

        import matplotlib.pyplot as plt

        plt.imshow(img, vmin=0, vmax=255, cmap='gray')

        plt.show()
