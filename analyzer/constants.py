from PyQt5.QtGui import QColor
from .model import ContainerRecord
GOOD_COLOR = QColor.fromRgb(0, 255, 0)
WARNING_COLOR = QColor.fromRgb(252, 132, 3)
BAD_COLOR = QColor.fromRgb(255, 0, 0)
ACTIVE_COLOR = QColor.fromRgb(245, 239, 66)
HOVER_COLOR = QColor.fromRgb(245, 66, 149)
BORDER_THICKNESS = 4
ACTIVE_BORDER_THICKNESS = 6
DETAIL_PANEL_WIDTH = 300
TITLE_COLOR = "#5e03fc"
TYPE_COLORS = {
    ContainerRecord.Type.CORRECT: GOOD_COLOR,
    ContainerRecord.Type.INCORRECT: BAD_COLOR,
    ContainerRecord.Type.MISSED: WARNING_COLOR
}