from typing import *
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtGui import QRawFont, QFont
from PyQt5.QtWidgets import QApplication

def filter_character_list(font_path: str, characters: List[str]):

    # load font
    app = QApplication.instance()

    if app is None:
        # if it does not exist then a QApplication is created
        app = QApplication([])
    else:
        print("QApplication already exists.")

    font = QRawFont(font_path, 10, QFont.PreferNoHinting)
    return [x for x in characters if font.supportsCharacter(x)]

def filter_character_list_advanced(font_path: str, characters: List[str]):
    filtered_characters = filter_character_list(font_path, characters)
    result = []
    # check characters one by one by drawing it onto the a mat
    W = 50
    H = 50
    font = ImageFont.truetype(font_path, W, encoding='utf-8')
    
    for ch in filtered_characters:
        img = Image.new('L', (W, H), 'white')
        draw = ImageDraw.Draw(img)

        offset_w, offset_h = font.getoffset(ch)
        w, h = draw.textsize(ch, font=font)
        pos = ((W-w-offset_w)/2, (H-h-offset_h)/2)

        # Draw
        draw.text(pos, ch, 'black', font=font)

        img_t = TF.to_tensor(img)

        #print(img_t.sum())

        if img_t.sum() != W * H:
            result.append(ch)
    
    return result


if __name__ == '__main__':
    import json
    
    with open("characters.json", 'r') as f:
        characters = json.loads(f.read())

    list = filter_character_list_advanced("fonts/TaipeiSansTCBeta-Regular.ttf", characters)
    print(list)
