from __future__ import annotations
import re
from typing import *

from PyQt5.QtCore import QAbstractItemModel, QModelIndex, QSortFilterProxyModel, Qt, QObject
from PyQt5.QtGui import QBrush

from .model import ContainerRecord
from .constants import TYPE_COLORS

class Node:
    def __init__(self, data: any, parent: Node = None) -> None:

        self.data = data
        self.children: List[Node] = []
        self.parent = parent
    
    def add_child(self, node: Node):
        self.children.append(node)
        node.parent = self
    
    def num_children(self):
        return len(self.children)
    
    def get_child(self, row: int):
        if row < len(self.children):
            return self.children[row]
        return None
    
    def get_parent(self):
        return self.parent
    
    def get_row(self):
        if self.parent is None:
            return 0
        return self.parent.children.index(self)

class QContainerRecordModel(QAbstractItemModel):

    def __init__(self, filename_to_record: Dict[str, List[ContainerRecord]]) -> None:
        super().__init__()
        # Tree structure: 
        # root (row = 0)
        #  |- filename 1 (row = 0)
        #         |- container 1 (row = 0)
        #         |- container 2 (row = 1)
        #  |- filename 2 (row = 1)
        #         |- container 1 (row = 0)
        #         |- container 2 (row = 1)
        #         |- container 3 (row = 2)

        # types
        # filename -> str
        # container -> ContainerRecord

        self.filename_to_records = filename_to_record

        # convert record into node tree
        self.root_node = Node(None)
        self.filename_to_node: Dict[str, Node] = {}

        for filename in self.filename_to_records.keys():

            if not filename in self.filename_to_node:
                node = Node(filename)
                self.filename_to_node[filename] = node
                self.root_node.add_child(node)

            node = self.filename_to_node[filename]

            for record in self.filename_to_records[filename]:
                node.add_child(Node(record))
            
        print(self.root_node)

    def index(self, row: int, column: int, parent: QModelIndex) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            # parent is root node
            parent_node = self.root_node
        else:
            parent_node: Node = parent.internalPointer() 
        
        child_node = parent_node.get_child(row)

        if child_node is not None:
            return self.createIndex(row, column, child_node)
        else:
            return QModelIndex()
    

    def parent(self, child: QModelIndex) -> QModelIndex:

        if not child.isValid():
            return QModelIndex()

        child_node: Node = child.internalPointer()

        if not child_node:
            return QModelIndex()

        parent_node = child_node.get_parent()

        if parent_node == self.root_node:
            return QModelIndex()
        
        return self.createIndex(parent_node.get_row(), 0, parent_node)
    
    def rowCount(self, parent: QModelIndex) -> int:
        if parent.column() > 0:
            return 0
        if not parent.isValid():
            parent_node = self.root_node
        else:
            parent_node: Node = parent.internalPointer()
        
        return parent_node.num_children()
    
    def columnCount(self, parent: QModelIndex) -> int:
        if parent.internalPointer() == self.root_node:
            return 0
        return 1
    
    def data(self, index: QModelIndex, role: int):

        if index.isValid():
        
            item: Node = index.internalPointer()

            if role == Qt.DisplayRole:

                if type(item.data) is str:
                    return item.data 

                if isinstance(item.data, ContainerRecord):
                    record: ContainerRecord = item.data
                    if record.type == ContainerRecord.Type.CORRECT:
                        return f"{record.index} {record.label}"
                    else:
                        return f"{record.index} {record.label} ► {record.predicted_label}"


            if role == Qt.ForegroundRole:
                brush = QBrush()

                if type(item.data) is str:
                    return None

                if isinstance(item.data, ContainerRecord):
                    record = item.data
                    brush.setColor(TYPE_COLORS[record.type])
                    return brush


    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return ""

class QContainerRecordFilterModel(QSortFilterProxyModel):

    def __init__(self) -> None:
        super().__init__()

        self.supported_modes = ("filename", "label", "predicted_label")
        self.mode = "filename"
        self.pattern: str = ""
        self.regexp: re.Pattern = None
    
    def set_pattern(self, pattern: str):
        if self.pattern != pattern:
            self.pattern = pattern
            self.regexp = re.compile(pattern)
            self.invalidateFilter()
    
    def set_mode(self, mode: str):
        if self.mode != mode:
            if mode not in self.supported_modes:
                raise ValueError(f"Mode should be one of those: {self.supported_modes}")
            self.mode = mode
            self.invalidateFilter()
    
    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:

        if len(self.pattern) == 0:
            return True
        
        model_index = self.sourceModel().index(source_row, 0, source_parent)
        
        item = model_index.internalPointer().data

        if self.mode == "filename":
            if type(item) is str:
                return True if self.regexp.search(item) else False
            return True
        


        if type(item) is str:
            # check child one by one
            found = False
            for row in range(self.sourceModel().rowCount(model_index)):
                child_model_index = self.sourceModel().index(row, 0, model_index)
                record = child_model_index.internalPointer().data
                if self.regexp.search(getattr(record, self.mode)):
                    found = True
                    break
            return found

        elif isinstance(item, ContainerRecord):
            record = item
            return True if self.regexp.search(getattr(record, self.mode)) else False