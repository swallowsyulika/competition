from typing import *
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


if __name__ == '__main__':
    import json
    
    with open("characters.json", 'r') as f:
        characters = json.loads(f.read())

    list = filter_character_list("fonts/TaipeiSansTCBeta-Regular.ttf", characters)
    print(list)
