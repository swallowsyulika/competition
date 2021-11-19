"""
Draw bonding boxes predicted by YOLOv4 onto images and save the images to output_dir.

Input: 
    * input_json: str
    * output_dir: str

Output: Images saved in output_dir
"""
import os
import json
import cv2
from tqdm import tqdm

import config
from utils import ensure_dir

# paths
input_json = config.eval_yolo_json_path
output_dir = config.eval_yolo_bb_drawn_path

ensure_dir(output_dir)

with open(input_json) as f:
    metadata = json.load(f)

    for data in tqdm(metadata):

        img_path = data["filename"]

        mat = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mat_h, mat_w, mat_c = mat.shape

        for obj in data["objects"]:

            coords = obj["relative_coordinates"]
            conf = obj["confidence"]

            cx = float(coords['center_x'])
            cy = float(coords['center_y'])
            _w = float(coords['width'])
            _h = float(coords['height'])

            w = int(mat_w * _w)
            h = int(mat_h * _h)

            x = int(cx * mat_w - w // 2)
            y = int(cy * mat_h - h // 2)

            

            cv2.rectangle(mat, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(mat, f"{int(conf * 100)}", (x, y), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255), fontScale=0.5)

        save_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, mat)
