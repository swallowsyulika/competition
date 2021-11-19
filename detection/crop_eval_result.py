"""
Crop bonding boxes predicted by YOLO v4 into images of characters.

Input: 
    * input_json: str
    * output_dir: str

Output: Images saved in output_dir
"""
import os
import json
import cv2
import numpy as np
from tqdm import tqdm

import config
from utils import ensure_dir, square_pad

input_json = config.eval_yolo_json_path
output_dir = config.eval_yolo_characters_path
conf_thresh = 0.9

ensure_dir(output_dir)


def find_direction(img_name):
    filename = os.path.splitext(img_name)[0]

    if filename.endswith("lr"):
        return "left_right"
    elif filename.endswith("tb"):
        return "top_bottom"

    return "unknown"

with open(input_json) as f:
    metadata = json.load(f)

    for data in tqdm(metadata):

        img_path = data["filename"]

        mat = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mat_h, mat_w, mat_c = mat.shape

        direction = find_direction(img_path)

        b_boxes = []

        for obj in data["objects"]:

            coords = obj["relative_coordinates"]
            conf = obj["confidence"]

            if conf < conf_thresh:
                continue

            cx = float(coords['center_x'])
            cy = float(coords['center_y'])
            _w = float(coords['width'])
            _h = float(coords['height'])

            w = int(mat_w * _w)
            h = int(mat_h * _h)

            x = int(cx * mat_w - w // 2)
            y = int(cy * mat_h - h // 2)

            b_boxes.append((x, y, w, h))

        if direction == "top_bottom":
            # left to right
            b_boxes_sorted = sorted(b_boxes, key=lambda x: x[0])
        elif direction == "left_right":
            # top to bottom 
            b_boxes_sorted = sorted(b_boxes, key=lambda x: x[1])
        else:
            print("[E] Something went wrong. Direction unknown!!!")
            print(img_path)
            cv2.imshow("win", mat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue

        for index, (x, y, w, h) in enumerate(b_boxes_sorted):
            x = max(x, 0)
            y = max(y, 0)

            if w == 0 or h == 0:
                continue
            save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_{index}.jpg")
            # print(x, y, w, h)
            # print(mat.shape)
            # print(mat[y:(y+h), x:(x+w)].shape)
            # print(x, y, w, h)
            # padded_img = square_pad()
            gray = cv2.cvtColor(mat[y:(y+h), x:(x+w)], cv2.COLOR_BGR2GRAY)
            cv2.imwrite(save_path, gray)