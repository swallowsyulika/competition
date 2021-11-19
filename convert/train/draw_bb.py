import os
import json
import cv2
import numpy as np
import random
import re
from PIL import ImageFont, Image, ImageDraw
from tqdm import tqdm
from config import train_path, train_bb_drawn_path, project_fonts_path
from utils import ensure_dir, is_chinese_only

# paths
workdir = train_path
img_dir = os.path.join(workdir, "img")
json_dir = os.path.join(workdir, "json")

output_dir = train_bb_drawn_path
ensure_dir(output_dir)

fontpath = os.path.join(project_fonts_path, "NotoSansCJK-Medium.ttc")

# some pre-defined colors used to draw bonding boxes
colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0),
]
# PIL font used to draw text onto the image
font = ImageFont.truetype(fontpath, 32)

for img in tqdm(os.listdir(img_dir)):
    if img.endswith(".jpg"):
        # read the image
        img_mat = cv2.imread(os.path.join(img_dir, img), cv2.IMREAD_UNCHANGED)
        # compose json filename
        json_name = img[:-3] + "json"

        # read the json file for labels 
        with open(os.path.join(json_dir, json_name), "r") as f:
            metadata = json.loads(f.read())
        
        for shape in metadata["shapes"]:

            label = shape["label"]

            # only draw labels that are composed of Chinese characters
            if not is_chinese_only(label):
                continue
            
            # randomly choose a color for each bonding box for clarity
            color = random.choice(colors)

            # print(shape)

            # draw bonding boxes onto the image 
            points = np.array(shape["points"], dtype=np.int32)
            cv2.polylines(img_mat, [points], True, color, 2)

            # draw label onto the image using PIL which allows the use of specified Chinese fonts
            img_pil = Image.fromarray(img_mat)
            draw = ImageDraw.Draw(img_pil)
            draw.text(points[0], label, font = font, fill = color)
            img_mat = np.array(img_pil)

        cv2.imwrite(os.path.join(output_dir, img), img_mat)
    else:
        print(f"[!] unknown file: {img}")
