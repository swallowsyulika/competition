import os
import csv
from utils.ensure_dir import ensure_dir
import numpy as np
import cv2
from tqdm import tqdm
from utils import four_point_transform, ensure_dir
from config import eval_img_path, eval_csv_path, eval_containers_path


# paths
img_dir = eval_img_path
csv_path = eval_csv_path
out_dir = eval_containers_path

ensure_dir(out_dir)

metadata = {}

with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    for filename, *points in reader:
        if filename not in metadata:
            metadata[filename] = []

        metadata[filename].append(
            np.array(points, dtype=np.float32).reshape(-1, 2))

# draw and save
imgs = [x for x in os.listdir(img_dir) if x.endswith(".jpg")]

for img_name in tqdm(imgs):
    img = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_UNCHANGED)
    for index, box in enumerate(metadata[img_name[:-4]]):
        cropped, _ = four_point_transform(img, box)
        h, w, c = cropped.shape
        side = max(h, w)
        padded_img = np.zeros((side, side, c), dtype=np.uint8)
        if w > h:
            pad_mode = "tb"
            y_begin = (side - h) // 2
            y_end = y_begin + h
            padded_img[y_begin:y_end, :, :] = cropped
        else:
            pad_mode = "lr"
            x_begin = (side - w) // 2
            x_end = x_begin + w
            padded_img[:, x_begin:x_end, :] = cropped

        # save the image 
        save_filename = f"{os.path.splitext(img_name)[0]}_{index}_{pad_mode}.jpg"
        cv2.imwrite(os.path.join(out_dir, save_filename), padded_img)
