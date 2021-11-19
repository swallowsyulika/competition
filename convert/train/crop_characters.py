import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from utils import four_point_transform, is_chinese_only, ensure_dir, square_pad
from config import train_path, train_characters_path

# paths
workdir = train_path

img_dir = os.path.join(workdir, "img")
json_dir = os.path.join(workdir, "json")
output_dir = train_characters_path
ensure_dir(output_dir)

for img in tqdm(os.listdir(img_dir)):
    if img.endswith(".jpg"):
        img_mat = cv2.imread(os.path.join(img_dir, img), cv2.IMREAD_UNCHANGED)
        json_name = img[:-3] + "json"

        with open(os.path.join(json_dir, json_name), "r") as f:
            metadata = json.loads(f.read())

        for shape in metadata["shapes"]:
            label = shape["label"]

            if not is_chinese_only(label) or len(label) > 1 or len(label) == 0 or "#" in label:
                continue

            points = shape["points"]
            result, _ = four_point_transform(img_mat, np.array(points, dtype=np.float32))

            dir_path = os.path.join(output_dir, label)

            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            padded_img = square_pad(result)
            candidate_name = f"{label}_{img}"

            while os.path.exists(os.path.join(dir_path, candidate_name)):
                candidate_name = f"{label}_{candidate_name}"

            gray = cv2.cvtColor(padded_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(dir_path, candidate_name), gray)

    else:
        print(f"[!] unknown file: {img}")
