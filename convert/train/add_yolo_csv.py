import os
import json
import numpy as np
import re
from tqdm import tqdm
from config import train_path
from utils import ensure_dir, is_chinese_only

workdir = train_path

json_dir = os.path.join(workdir, "json")
output_dir = os.path.join(workdir, "csv")
ensure_dir(output_dir)

obj_class = 0

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ignore_pat = re.compile("[A-z0-9]+")

for json_file in tqdm(os.listdir(json_dir)):
    if json_file.endswith(".json"):

        with open(os.path.join(json_dir, json_file), "r") as f:
            metadata = json.loads(f.read())
        
        w, h = int(metadata['imageWidth']), int(metadata['imageHeight'])

        with open(os.path.join(output_dir, f"{json_file[:-5]}.csv"), 'w') as csv_file:

            for shape in metadata["shapes"]:
                label = shape["label"]

                if is_chinese_only(label) and len(label) == 1 and not '#' in label:
                    points = shape["points"]
                    points = np.array(points)
                    center = np.mean(points, axis=0)

                    x, y = center[0].item(), center[1].item()

                    points_x = points[:, 0]
                    points_y = points[:, 1]

                    width = (points_x.max() - points_x.min()).item()
                    height = (points_y.max() - points_y.min()).item()

                    #<x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
                    bbox_x = x/w
                    bbox_y = y/h
                    bbox_w = width / w
                    bbox_h = height/ h

                    csv_file.write(f"{obj_class} {bbox_x} {bbox_y} {bbox_w} {bbox_h}\n")
            
    else:
        print(f"[!] unknown file: {json_file}")
