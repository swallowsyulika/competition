import csv
from enum import Enum, auto
from sys import hexversion
from typing import *
class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

class ContainerRecord:

    class Type(Enum):
        CORRECT = auto()
        INCORRECT = auto()
        MISSED = auto()

    def __init__(self,
        filename: str,
        points: List[Point],
        label: str,
        predicted_label: str,
        index: int,

    ) -> None:
        self.filename = filename
        self.points = points
        self.label = label
        self.predicted_label = predicted_label
        self.index = index
        self.type: ContainerRecord.Type = None

        if self.label == self.predicted_label:
            self.type = ContainerRecord.Type.CORRECT
        elif self.predicted_label == "###":
            self.type = ContainerRecord.Type.MISSED
        else:
            self.type = ContainerRecord.Type.INCORRECT

filename_to_containers: Dict[str, List[ContainerRecord]] = {}

def parse_csv(filepath: str, predicted_filepath):

    records: List[ContainerRecord] = []

    with open(filepath, 'r', encoding='utf-8') as f_ground, open(predicted_filepath, 'r', encoding='utf-8') as f_predicted:
        ground_data = list(csv.reader(f_ground))
        predicted_data = list(csv.reader(f_predicted))

        last_filename = ""
        index = 0
        
        
        for i in range(len(ground_data)):

            filename, x1, y1, x2, y2, x3, y3, x4, y4, label = ground_data[i]
            p_filename, p_x1, p_y1, p_x2, p_y2, p_x3, p_y3, p_x4, p_y4, p_label = predicted_data[i]

            if filename != p_filename:
                print(filename)
                print(p_filename)
                raise ValueError("Two csv files does not match.")


            if last_filename != filename:
                index = 0
            else:
                index += 1
            
            record = ContainerRecord(
                filename,
                [
                    Point(int(x1), int(y1)),
                    Point(int(x2), int(y2)),
                    Point(int(x3), int(y3)),
                    Point(int(x4), int(y4)),
                ],
                label,
                p_label,
                index
            )

            records.append(record)

            if filename not in filename_to_containers:
                filename_to_containers[filename] = []

            filename_to_containers[filename].append(record)                

            last_filename = filename
    
    return records

class CharacterRecord:

    def __init__(self, x: int, y: int, w: int, h: int, conf: float) -> None:

        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.confidence = conf