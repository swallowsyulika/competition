import os
import re
import sys
from typing import *

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .model import *
from .constants import *
from .dark_theme import palette
from .widgets import *
from .qt_model import *
from .utils import *

import config

if config.mode != "pseudo_eval":
    print("[E] This module is only available when in pseudo_eval mode.")
    sys.exit()

result_csv = config.eval_recognition_csv_path
ground_csv = config.pseudo_eval_ground_csv_path

imgs_path = config.eval_img_path
cleaner_path = config.eval_cleaned_path
container_path = config.eval_containers_path

cleaned_characters_path = config.eval_cleaned_path



class ContainerGraphicsItem(QGraphicsPolygonItem):
    def __init__(self, record: ContainerRecord, click_handler: Callable[[ContainerRecord], None], active: bool = False):
        super().__init__()

        # states
        poly = QPolygonF([QPoint(p.x, p.y) for p in record.points])
        self.setPolygon(poly)

        self.record = record

        self.setAcceptHoverEvents(True)

        self.is_hovering = False

        # handlers
        self.click_handler = click_handler

        # set border style
        self.regular_pen = QPen()
        self.regular_pen.setColor(TYPE_COLORS[self.record.type])
        self.regular_pen.setWidth(
            ACTIVE_BORDER_THICKNESS if active else BORDER_THICKNESS)

        # set border style when hover
        self.hover_pen = QPen()
        self.hover_pen.setColor(HOVER_COLOR)
        self.hover_pen.setWidth(BORDER_THICKNESS)

        # set fill style
        self.regular_brush = QBrush()

        if active:
            self.regular_brush.setColor(
                change_alpha(self.regular_pen.color(), 150))
        else:
            self.regular_brush.setColor(QColor.fromRgb(0, 0, 0, 0))

        self.regular_brush.setStyle(Qt.SolidPattern)

        # set fill style when hover
        self.hover_brush = QBrush()
        self.hover_brush.setColor(change_alpha(HOVER_COLOR, 100))
        self.hover_brush.setStyle(Qt.SolidPattern)

        self.setPen(self.regular_pen)
        self.setBrush(self.regular_brush)


    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.setPen(self.hover_pen)
        self.setBrush(self.hover_brush)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.setPen(self.regular_pen)
        self.setBrush(self.regular_brush)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pass

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.click_handler(self.record)

# Subclass QMainWindow to customize your application's main window


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # states
        self.records: List[ContainerRecord] = None
        self.filenames: List[str] = None
        self.records_model: QContainerRecordModel = None
        self.filtered_records_model = QContainerRecordFilterModel = None
        self.current_filename_index: int = None
        self.selection_model: QItemSelectionModel = None

        # setup window
        self.setWindowTitle("Analyzer")
        self.setGeometry(0, 0, 1280, 720)

        # setup UI

        self.main_layout = QHBoxLayout()

        self.splitter = QSplitter()
        # container list pane
        self.container_tree_layout = QVBoxLayout()
        self.container_tree_layout.setContentsMargins(0, 0, 0, 0)
        # search input
        self.search_layout = QHBoxLayout()

        self.container_list_search_lineedit = QLineEdit()
        self.container_list_search_lineedit.setPlaceholderText("Search for keyword...")
        self.search_layout.addWidget(self.container_list_search_lineedit)

        self.search_type_combo = QComboBox()
        self.search_type_combo.addItems(["filename", "label", "predicted_label"])
        self.search_layout.addWidget(self.search_type_combo)

        self.container_tree_layout.addLayout(self.search_layout)
        # container list
        self.container_tree_view_widget = QTreeView()
        self.container_tree_view_widget.setHeaderHidden(True)
        self.container_tree_layout.addWidget(self.container_tree_view_widget)
        # buttons
        self.container_tree_buttons = QGridLayout()

        self.expand_all_btn = QPushButton("Expand All")
        self.container_tree_buttons.addWidget(self.expand_all_btn, 0, 0)
        self.collapse_all_btn = QPushButton("Collapse All")
        self.container_tree_buttons.addWidget(self.collapse_all_btn, 0, 1)

        self.prev_sample_btn = QPushButton("Prev Sample")
        self.container_tree_buttons.addWidget(self.prev_sample_btn, 1, 0)
        self.next_sample_btn = QPushButton("Next Sample")
        self.container_tree_buttons.addWidget(self.next_sample_btn, 1, 1)

        self.prev_img_btn = QPushButton("Prev Image")
        self.container_tree_buttons.addWidget(self.prev_img_btn, 2, 0)
        self.next_img_btn = QPushButton("Next Image")
        self.container_tree_buttons.addWidget(self.next_img_btn, 2, 1)
        self.next_error_btn = QPushButton("Next Error")
        self.container_tree_buttons.addWidget(self.next_error_btn, 3, 0)
        self.next_missed_btn = QPushButton("Next Missed")
        self.container_tree_buttons.addWidget(self.next_missed_btn, 3, 1)

        self.container_tree_layout.addLayout(self.container_tree_buttons)

        self.container_list_panel_widget = QWidget()
        self.container_list_panel_widget.setLayout(self.container_tree_layout)
        self.splitter.addWidget(self.container_list_panel_widget)

        # graphics
        self.graphics_scene = QGraphicsScene()
        self.graphics_view = WheelZoomGraphicsView()
        self.graphics_view.setRenderHints(
            QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.graphics_view.setScene(self.graphics_scene)
        self.splitter.addWidget(self.graphics_view)

        self.main_layout.addWidget(self.splitter)

        # details
        self.details_panel = DetailsPanel()
        self.main_layout.addWidget(self.details_panel)

        # status bar
        self.status_bar = self.statusBar()

        # Set the central widget of the Window.
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)

        self.setCentralWidget(self.main_widget)

        self.handle_event()
        self.load_data()

    def handle_event(self):
        """
        Connect handler functions to signals of UI components
        """
        self.search_type_combo.currentTextChanged.connect(self.handle_mode_change)
        self.container_list_search_lineedit.textEdited.connect(self.handle_pattern_change)
        self.expand_all_btn.clicked.connect(lambda: self.container_tree_view_widget.expandAll())
        self.collapse_all_btn.clicked.connect(lambda: self.container_tree_view_widget.collapseAll())
        self.prev_sample_btn.clicked.connect(self.handle_prev_sample)
        self.next_sample_btn.clicked.connect(self.handle_next_sample)

        self.prev_img_btn.clicked.connect(lambda: self.change_img(-1))
        self.next_img_btn.clicked.connect(lambda: self.change_img(1))

        self.next_error_btn.clicked.connect(self.handle_next_incorrect)
        self.next_missed_btn.clicked.connect(self.handle_next_missed)

        self.graphics_view.coordsChanged.connect(self.handle_coords_change)

    def load_data(self):

        self.records = parse_csv(ground_csv, result_csv)
        self.records_model = QContainerRecordModel(filename_to_containers)
        self.filtered_records_model = QContainerRecordFilterModel()
        self.filtered_records_model.setSourceModel(self.records_model)
        self.filenames = list(filename_to_containers.keys())
        self.container_tree_view_widget.setModel(self.filtered_records_model)
        self.selection_model = self.container_tree_view_widget.selectionModel()
        self.container_tree_view_widget.expandAll()
        self.selection_model.selectionChanged.connect(self.handle_selection_change)
        self.set_model_index(self.filtered_records_model.index(0, 0, QModelIndex()))
    
    def handle_coords_change(self, x, y):
        item = self.graphics_scene.itemAt(float(x), float(y), QTransform())
        if isinstance(item, QGraphicsPixmapItem):
            self.status_bar.showMessage(f"{x, y}")
        elif isinstance(item, ContainerGraphicsItem):
            msg = f"{x, y} |" 
            record: ContainerRecord = item.record

            for index, point in enumerate(record.points):
                msg += f" p{index+1}: {point.x, point.y}"
            
            self.status_bar.showMessage(msg)
        else:
            self.status_bar.showMessage("")


    def handle_mode_change(self, mode: str):
        self.filtered_records_model.set_mode(mode) 
        self.container_tree_view_widget.expandAll()
    def handle_pattern_change(self, pattern: str):
        self.filtered_records_model.set_pattern(pattern)
        self.container_tree_view_widget.expandAll()
    
    def handle_selection_change(self, selected: QItemSelection, _):
        # we need to get ModelIndex of source model in order to get object stored in internalPointer()
        indexes = selected.indexes()
        if len(indexes) > 0:
            item = self.filtered_records_model.mapToSource(indexes[0]).internalPointer().data
            self.load_item(item)

    def _get_current_row(self):
        return self.container_tree_view_widget.currentIndex().row()

    def handle_next_sample(self):
        model_index = self._find_next(lambda _: True)
        if model_index is not None:
            self.set_model_index(model_index)

    def handle_prev_sample(self):
        model_index = self._find_next(lambda _: True, -1)
        if model_index is not None:
            self.set_model_index(model_index)
    

    def _find_next(self, criterion, direction: int = 1):

        if direction not in (1, -1):
            raise ValueError("direction should be 1 (forward) or -1 (backward).")
        
        def next_index(model_index: QModelIndex):
            item = self.filtered_records_model.mapToSource(model_index).internalPointer().data
            print(item)

            if type(item) is str:
                # filename level
                if direction == 1:
                    # return first child
                    return self.filtered_records_model.index(0, 0, model_index)
                else:
                    # return last child
                    num_rows = self.filtered_records_model.rowCount(model_index)
                    return self.filtered_records_model.index(num_rows - 1, 0, model_index)

            elif isinstance(item, ContainerRecord):
                # ContainerRecord level:
                # return next ContainerRecord if any
                # or return next filename model index

                parent_model_index = self.filtered_records_model.parent(model_index)
                next_model_index = self.filtered_records_model.index(model_index.row() + direction, 0, parent_model_index)

                if next_model_index.isValid():
                    return next_model_index
                
                next_model_index = self.filtered_records_model.index(parent_model_index.row() + direction, 0, QModelIndex())
                if next_model_index.isValid():
                    return next_index(next_model_index)
                
                return None

        model_index = self.get_current_model_index()

        current_model_index = model_index

        while True:
            current_model_index = next_index(current_model_index)
            if current_model_index is None:
                break
            if criterion(self.filtered_records_model.mapToSource(current_model_index).internalPointer().data):
                return current_model_index
        return None



    def get_current_model_index(self):
        return self.selection_model.selectedIndexes()[0] 
    
    def set_model_index(self, index: QModelIndex):
        print("set model index called!")
        self.selection_model.select(index, QItemSelectionModel.ClearAndSelect)
        self.container_tree_view_widget.scrollTo(index)

    def change_img(self, direction: int = 1):

        # find current index
        model_index = self.get_current_model_index()

        item = self.filtered_records_model.mapToSource(model_index).internalPointer().data
        
        # get to filename level if we're at ContainerRecord level
        if isinstance(item, ContainerRecord):
            model_index = self.filtered_records_model.parent(model_index)
        
        new_row = model_index.row() + direction
        new_model_index = self.filtered_records_model.sibling(new_row, 0, model_index)

        if new_model_index.isValid():
            self.set_model_index(new_model_index)

        



    def handle_next_incorrect(self):
        def criterion(nextRecord: ContainerRecord):
            return (nextRecord.label != nextRecord.predicted_label) and nextRecord.predicted_label != "###"

        model_index = self._find_next(criterion)
        if model_index is not None:
            self.set_model_index(model_index)

    def handle_next_missed(self):
        def criterion(nextRecord: ContainerRecord):
            return nextRecord.predicted_label == "###"

        model_index = self._find_next(criterion)
        if model_index is not None:
            self.set_model_index(model_index)

    def handle_container_clicked(self, record: ContainerRecord):
        index = self.filenames.index(record.filename)
        model_index = self.records_model.index(index, 0, QModelIndex())

        img_records = filename_to_containers[record.filename]
        index = img_records.index(record)

        model_index = self.records_model.index(index, 0, model_index)

        while True:
            filtered_model_index = self.filtered_records_model.mapFromSource(model_index)
            
            if filtered_model_index.isValid():
                break
            else:
                self.filtered_records_model.set_pattern("")
                self.container_list_search_lineedit.setText("")
                self.container_tree_view_widget.expandAll()

        self.set_model_index(filtered_model_index)


    def get_cleaner_img_paths(self, record: ContainerRecord):
        img_path_begin = f"{record.filename}_{record.index}_"
        pattern = re.compile(f"{img_path_begin}((lr)|(tb))_([0-9]+).jpg")

        img_paths = sorted([p for p in os.listdir(cleaner_path) if p.startswith(
            img_path_begin)], key=lambda x: int(pattern.search(x).group(4)))
        return [os.path.join(cleaner_path, x) for x in img_paths]

    def get_container_img_path(self, record: ContainerRecord):
        img_path_begin = f"{record.filename}_{record.index}_"

        path = os.path.join(container_path, f"{img_path_begin}tb.jpg")
        path2 = os.path.join(container_path, f"{img_path_begin}lr.jpg")
        if os.path.exists(path):
            return path
        elif os.path.exists(path2):
            return path2
        else:
            raise FileNotFoundError("Unable to find container image file.")
    

    def load_item(self, item: Union[str, ContainerRecord]):

        img_records: List[ContainerRecord] = None        

        if isinstance(item, ContainerRecord):
            # item is ContainerRecord
            record = item
            filename = record.filename
            img_path = os.path.join(imgs_path, record.filename)
            img_records = filename_to_containers[record.filename]
            has_active_item = True

        elif type(item) is str:
            # item is filename
            filename = item
            img_path = os.path.join(imgs_path, filename)
            img_records = filename_to_containers[filename]
            has_active_item = False


        # Load specified image to the graphics scene.
        image = QImage()
        image.load(img_path)
        pixmap = QPixmap.fromImage(image)

        self.graphics_scene.clear()

        pixmap_item = self.graphics_scene.addPixmap(pixmap)

        # draw containers poly

        for r in img_records:
            # create polygon
            self.graphics_scene.addItem(
                ContainerGraphicsItem(
                r, 
                lambda r: self.handle_container_clicked(r),
                (r == record) if has_active_item else False
                )
            )

        # reset scene rect so that scroll bar won't appear
        self.graphics_scene.setSceneRect(
            self.graphics_scene.itemsBoundingRect())

        if self.current_filename_index is None or filename != self.filenames[self.current_filename_index]:
            # fit the image to the view
            self.graphics_view.fitInView(pixmap_item, Qt.KeepAspectRatio)

        # load cleaner pics
        if has_active_item:
            cleaned_img_paths = self.get_cleaner_img_paths(record)
            container_img_path = self.get_container_img_path(record)

            self.details_panel.set_record_details(
                record.label, record.predicted_label, container_img_path, cleaned_img_paths)
        else:
            num_correct = 0
            num_incorrect = 0
            num_missed = 0

            for record in img_records:
                if record.type == ContainerRecord.Type.CORRECT:
                    num_correct += 1
                elif record.type == ContainerRecord.Type.INCORRECT:
                    num_incorrect += 1
                elif record.type == ContainerRecord.Type.MISSED:
                    num_missed += 1

            self.details_panel.set_image_details(num_correct, num_incorrect, num_missed)
        
        self.current_filename_index = self.filenames.index(filename)




app = QApplication(sys.argv)

# Apply dark theme using palette
# Force the style to be the same on all OSs
app.setStyle("Fusion")
app.setPalette(palette)


window = MainWindow()
window.show()

app.exec()
