from PyQt5.QtGui import QColor

def change_alpha(color: QColor, alpha: int):
    """
    Return the given color with specified alpha value
    """
    r_color = QColor(color)
    r_color.setAlpha(alpha)
    return r_color