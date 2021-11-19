from typing import *
import json
import os

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from config import eval_yolo_json_path

from .constants import *
from .model import CharacterRecord
from .utils import *


class DetailItem(QWidget):

    def __init__(self, name: str):
        super().__init__()
        # states
        self.expanded = True
        # button
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.name = name
        self.title_button = QPushButton(self._get_expanded_title())
        self.title_button.setStyleSheet(
            f"background: {TITLE_COLOR}; font-size: 16px; font-weight: 400; text-align: left;")
        self.title_button.clicked.connect(self.toggle)
        self._layout.addWidget(self.title_button)

        self.content_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignHCenter)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.content_widget.setLayout(self.main_layout)
        self._layout.addWidget(self.content_widget)
        # self.setStyleSheet("background: red;")
        self.setLayout(self._layout)

        self.animator = QPropertyAnimation(
            self.content_widget, b"maximumHeight")
        self.animator.setDuration(250)
        self.animator.setEasingCurve(QEasingCurve.InOutQuad)
        self.animator.finished.connect(self._handle_animation_finished)

    def _get_expanded_title(self):
        return f"  ▼ {self.name}"

    def _get_collapsed_title(self):
        return f"  ▲ {self.name}"

    def _handle_animation_finished(self):
        """
        It's necessary to set maximumHeight back to QWIDGETSIZE_MAX so that
        the height of content_widget won't be constrained when switching to
        other samples.
        """
        if self.expanded:
            self.content_widget.setMaximumHeight(QWIDGETSIZE_MAX)

    def expand(self):
        self.animator.setStartValue(0)
        self.animator.setEndValue(self.content_widget.sizeHint().height())
        self.animator.start()
        self.title_button.setText(self._get_expanded_title())
        self.expanded = True

    def collapse(self):
        self.animator.setStartValue(self.content_widget.sizeHint().height())
        self.animator.setEndValue(0)
        self.animator.start()
        self.title_button.setText(self._get_collapsed_title())
        self.expanded = False

    def toggle(self):
        if self.expanded:
            self.collapse()
        else:
            self.expand()

    def hide(self):
        self.setMaximumHeight(0)

    def show(self):
        self.setMaximumHeight(QWIDGETSIZE_MAX)


class ContainerTextViewer(DetailItem):
    def __init__(self):
        super().__init__("Text")
        self.font = QFont()
        self.font.setPointSize(20)

        self.ground_label = QLabel()
        self.ground_label.setFont(self.font)
        self.main_layout.addWidget(self.ground_label)
        self.predicted_label = QLabel()
        self.predicted_label.setFont(self.font)
        self.main_layout.addWidget(self.predicted_label)

    def set_text(self, ground: str, predicted: str):
        self.ground_label.setText(ground)
        self.predicted_label.setText(predicted)

        palette = QPalette()
        if ground == predicted:
            palette.setColor(self.predicted_label.foregroundRole(), GOOD_COLOR)
        elif predicted == "###":
            palette.setColor(
                self.predicted_label.foregroundRole(), WARNING_COLOR)
        else:
            palette.setColor(self.predicted_label.foregroundRole(), BAD_COLOR)

        self.predicted_label.setPalette(palette)


class TransformedContainerViewer(DetailItem):

    class BondingBoxGraphicsItem(QGraphicsRectItem):

        NORMAL_COLOR = QColor.fromRgb(213, 29, 219)
        HOVER_COLOR = QColor.fromRgb(34, 205, 240)

        def __init__(self, record: CharacterRecord):

            super().__init__(
                0,
                0,
                float(record.width),
                float(record.height))
            
            pen_width = record.width * 0.05
            self.setPos(record.x, record.y)
            # setup brush
            self.normal_brush = QBrush()
            self.normal_brush.setColor(QColor.fromRgb(0, 0, 0, 0))
            self.normal_brush.setStyle(Qt.SolidPattern)
            self.setBrush(self.normal_brush)

            self.hover_brush = QBrush()
            self.hover_brush.setColor(change_alpha(self.HOVER_COLOR, 100))
            self.hover_brush.setStyle(Qt.SolidPattern)

            # setup pen
            self.normal_pen = QPen()
            self.normal_pen.setColor(change_alpha(self.NORMAL_COLOR, 100))
            self.normal_pen.setWidthF(pen_width)

            self.hover_pen = QPen()
            self.hover_pen.setColor(self.HOVER_COLOR)
            self.hover_pen.setWidthF(pen_width)
            self.setPen(self.normal_pen)

            # setup confidence text
            self.text_brush = QBrush()
            self.text_brush.setColor(self.NORMAL_COLOR)
            self.text_brush.setStyle(Qt.SolidPattern)
            self.conf_text = QGraphicsSimpleTextItem()
            self.font = QFont()
            self.font.setPointSizeF(record.width * 0.3)
            self.font.setBold(True)
            self.conf_text.setFont(self.font)
            self.conf_text.setText(f"{record.confidence * 100: .0f}")
            self.conf_text.setBrush(self.text_brush)
            self.conf_text.setParentItem(self)
            bonding_rect = self.conf_text.boundingRect()
            self.conf_text.setPos((record.width - bonding_rect.width()) / 2, (record.height - bonding_rect.height()) / 2)
            self.conf_text.setVisible(False)

            # listen hover events
            self.setAcceptHoverEvents(True)

        def hoverEnterEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
            self.setBrush(self.hover_brush)
            self.setPen(self.hover_pen)
            self.conf_text.setVisible(True)
            return super().hoverEnterEvent(event)

        def hoverLeaveEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
            self.setBrush(self.normal_brush)
            self.setPen(self.normal_pen)
            self.conf_text.setVisible(False)
            return super().hoverLeaveEvent(event)

    def __init__(self):
        super().__init__("Transformed Container")

        # self.image_label = QLabel()

        self.graphics_scene = QGraphicsScene()
        self.graphics_view = QGraphicsView()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setMinimumSize(
            DETAIL_PANEL_WIDTH, DETAIL_PANEL_WIDTH)

        self.main_layout.addWidget(self.graphics_view)

        # self.main_layout.addWidget(self.image_label)
        self.pixmap = None

        self._cache_detection_data()

    def set_image(self, img_path: str):
        # cleanup graphics scene
        self.graphics_scene.clear()
        # load image
        qimg = QImage()
        qimg.load(img_path)
        self.pixmap = QPixmap.fromImage(qimg)
        pixmap_item = self.graphics_scene.addPixmap(self.pixmap)

        # load detection result
        records = self.get_detection_data(
            img_path, qimg.width(), qimg.height())

        for record in records:
            self.graphics_scene.addItem(self.BondingBoxGraphicsItem(record))

        itemsBoundingRect = self.graphics_scene.itemsBoundingRect()
        self.graphics_scene.setSceneRect(itemsBoundingRect)
        self.graphics_view.fitInView(itemsBoundingRect, Qt.KeepAspectRatio)

    def _cache_detection_data(self):

        # load yolo detection result
        with open(eval_yolo_json_path) as f:
            yolo_json_obj = json.load(f)

        self.yolo_container_path_to_objs = {
            os.path.basename(item["filename"]): item["objects"] for item in yolo_json_obj}

    def get_detection_data(self, container_img_path: str, img_width: int, img_height: int) -> List[ContainerRecord]:
        return self._objs_to_character_records(
            self.yolo_container_path_to_objs[os.path.basename(container_img_path)], img_width, img_height)

    def _objs_to_character_records(self, objs, img_width: int, img_height: int):
        """
        Convert YOLO v4 detection result "objects" field into list of CharacterRecord.
        """
        records: List[CharacterRecord] = []

        for obj in objs:

            coords = obj["relative_coordinates"]

            conf = obj["confidence"]

            cx = float(coords['center_x'])
            cy = float(coords['center_y'])
            _w = float(coords['width'])
            _h = float(coords['height'])

            w = int(img_width * _w)
            h = int(img_height * _h)

            x = int(cx * img_width - w // 2)
            y = int(cy * img_height - h // 2)

            records.append(CharacterRecord(x, y, w, h, conf))

        return records


class CleanedImagesViewer(DetailItem):

    def __init__(self):
        super().__init__("Cleaned Images")

        self.image_labels: List[QLabel] = []

    def set_images(self, img_paths: List[str]):
        # delete old images
        for label in self.image_labels:
            self.main_layout.removeWidget(label)
            label.setParent(None)
            label.deleteLater()

        self.image_labels.clear()

        # put new images
        for img_path in img_paths:
            print(img_path)
            qimg = QImage()
            qimg.load(img_path)
            pixmap = QPixmap.fromImage(qimg)
            label = QLabel()
            # label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            label.setPixmap(pixmap)
            print(label.sizeHint())
            self.image_labels.append(label)
            self.main_layout.addWidget(label)


class ImageStats(DetailItem):
    def __init__(self):
        super().__init__("Stats")

        font_size = "20px"

        self.correct_label = QLabel()
        self.correct_label.setStyleSheet(
            f"font-size: {font_size}; color: {TYPE_COLORS[ContainerRecord.Type.CORRECT].name()}")
        self.main_layout.addWidget(self.correct_label)

        self.incorrect_label = QLabel()
        self.incorrect_label.setStyleSheet(
            f"font-size: {font_size}; color: {TYPE_COLORS[ContainerRecord.Type.INCORRECT].name()}")
        self.main_layout.addWidget(self.incorrect_label)

        self.missed_label = QLabel()
        self.missed_label.setStyleSheet(
            f"font-size: {font_size}; color: {TYPE_COLORS[ContainerRecord.Type.MISSED].name()}")
        self.main_layout.addWidget(self.missed_label)

    def _get_percent(self, portion: int, total: int):
        return f"{round((portion / total) * 100, 1)}%"

    def set_stats(self, num_correct: int, num_incorrect: int, num_missed: int):

        total = num_correct + num_incorrect + num_missed
        self.correct_label.setText(
            f"correct: {num_correct} ({self._get_percent(num_correct, total)})")
        self.missed_label.setText(
            f"missed: {num_missed} ({self._get_percent(num_missed, total)})")
        self.incorrect_label.setText(
            f"incorrect: {num_incorrect} ({self._get_percent(num_incorrect, total)})")


class DetailsPanel(QScrollArea):

    def __init__(self):
        super().__init__()

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        pad = 20
        self.setMinimumWidth(DETAIL_PANEL_WIDTH + pad)
        self.setMaximumWidth(DETAIL_PANEL_WIDTH + pad)
        self.setWidgetResizable(True)

        self.record_items: List[DetailItem] = []
        self.image_items: List[DetailItem] = []

        # stats
        self.stats_viewer = ImageStats()
        self.main_layout.addWidget(self.stats_viewer)
        self.image_items.append(self.stats_viewer)

        # text
        self.text_viewer = ContainerTextViewer()
        self.main_layout.addWidget(self.text_viewer)
        self.record_items.append(self.text_viewer)
        # transformed container
        self.transformed_container_viewer = TransformedContainerViewer()
        self.main_layout.addWidget(self.transformed_container_viewer)
        self.record_items.append(self.transformed_container_viewer)
        # cleaner
        self.cleaned_imgs_viewer = CleanedImagesViewer()
        self.main_layout.addWidget(self.cleaned_imgs_viewer)
        self.record_items.append(self.cleaned_imgs_viewer)

        self.main_widget = QWidget()
        self.main_widget.setContentsMargins(0, 0, 0, 0)
        self.main_widget.setLayout(self.main_layout)

        self.main_layout.addStretch()

        self.setWidget(self.main_widget)

    def set_record_details(self, ground_text: str, predicted_text: str, container_path: str, cleaned_img_paths: List[str]):
        for item in self.image_items:
            item.hide()

        for item in self.record_items:
            item.show()

        self.text_viewer.set_text(ground_text, predicted_text)
        self.transformed_container_viewer.set_image(container_path)
        self.cleaned_imgs_viewer.set_images(cleaned_img_paths)

        if len(cleaned_img_paths) == 0:
            self.cleaned_imgs_viewer.hide()
        else:
            self.cleaned_imgs_viewer.show()

        print(self.main_widget.sizeHint())

        self.update()

    def set_image_details(self, num_correct: int, num_incorrect: int, num_missed: int):
        # hide all
        for item in self.record_items:
            item.hide()
        for item in self.image_items:
            item.show()

        self.stats_viewer.set_stats(num_correct, num_incorrect, num_missed)


class WheelZoomGraphicsView(QGraphicsView):

    coordsChanged = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.zoom_scale = 1.4
        self.zoom_out_scale = 1 / self.zoom_scale

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        if event.angleDelta().y() > 0:
            self.scale(self.zoom_scale, self.zoom_scale)
        else:
            self.scale(self.zoom_out_scale, self.zoom_out_scale)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        clicked_item = self.itemAt(event.pos())
        if isinstance(clicked_item, QGraphicsPixmapItem):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.setDragMode(QGraphicsView.NoDrag)
        return super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        point: QPointF = self.mapToScene(event.pos())
        self.coordsChanged.emit(int(point.x()), int(point.y()))
        return super().mouseMoveEvent(event)
