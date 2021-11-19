import os
import random
import csv
import numpy as np
import cv2
from tqdm import tqdm
from config import eval_img_path, eval_csv_path, eval_containers_drawn_path
from utils import ensure_dir

# paths
img_dir = eval_img_path
csv_path = eval_csv_path
out_dir = eval_containers_drawn_path
ensure_dir(out_dir)

metadata = {}

# some pre-defined colors used to draw bonding boxes
colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0),
]

with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    for filename, *points in reader:
        if filename not in metadata:
            metadata[filename] = []

        metadata[filename].append(
            np.array(points, dtype=np.int32).reshape(-1, 2))

# draw and save
imgs = [x for x in os.listdir(img_dir) if x.endswith(".jpg")]

for img_name in tqdm(imgs):
    img = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_UNCHANGED)
    for index, points in enumerate(metadata[img_name[:-4]]):
        color = random.choice(colors)
        cv2.polylines(img, [points], True, color, thickness=2)
        cv2.imwrite(os.path.join(out_dir, img_name), img)
