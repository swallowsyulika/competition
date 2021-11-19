import random
import re
import os
import json
import shutil
from utils.random_highlight_color import random_highlight_color

import numpy as np
import cv2
from tqdm import tqdm

from utils import ensure_dir, is_chinese_only, ensure_file, random_highlight_color
import config

# paths
data_dir = config.train_path

img_dir = os.path.join(data_dir, "img")
json_dir = os.path.join(data_dir, "json")

output_img_dir = config.pseudo_eval_img_path
outupt_csv_path = config.pseudo_eval_csv_path
output_ground_csv_path = config.pseudo_eval_ground_csv_path

ensure_dir(output_img_dir)
ensure_file(outupt_csv_path)
ensure_file(output_ground_csv_path)

def character_in_container(character: np.ndarray, container: np.ndarray, thresh=0.7):

    character = character.astype(np.int32)
    container = container.astype(np.int32)

    ch_xs = character[:, 0]
    ch_ys = character[:, 1]

    c_xs = container[:, 0]
    c_ys = container[:, 1]

    xs = np.concatenate((ch_xs, c_xs), axis=0)
    ys = np.concatenate((ch_ys, c_ys), axis=0)

    min_x = np.min(xs)
    min_y = np.min(ys)

    max_x = np.max(xs)
    max_y = np.max(ys)

    mat_w = max_x - min_x
    mat_h = max_y - min_y

    container_mask = np.zeros((mat_h, mat_w), dtype=np.uint8)
    character_mask = np.zeros((mat_h, mat_w), dtype=np.uint8)

    def draw_points_to_mask(points, mask):
        # print(mask.shape)
        points_norm = points - np.array([min_x, min_y])
        # print(points_norm)
        cv2.fillPoly(mask, [points_norm], 255)

    # draw container mask
    draw_points_to_mask(container, container_mask)

    # draw character mask
    draw_points_to_mask(character, character_mask)

    container_mask_binary = container_mask > 0
    character_mask_binary = character_mask > 0

    return np.sum(container_mask_binary & character_mask_binary) / np.sum(character_mask_binary) > thresh


def polys_to_rects(points: np.ndarray, remove_negative=True):

    if remove_negative:
        points = np.array(points, copy=True)
        points = points.clip(min=0)

    rects = []

    n_rects, _, _ = points.shape

    for i in range(n_rects):

        xs = points[i, :, 0]
        ys = points[i, :, 1]

        cx = xs.mean()
        cy = ys.mean()

        x_min = xs.min()
        x_max = xs.max()

        y_min = ys.min()
        y_max = ys.max()

        w = x_max - x_min
        h = y_max - y_min

        rects.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])

    return rects



metadata = {}  # [filename] => container boxes
character_boxes = {}  # [filename] => character boxes


for json_file in os.listdir(json_dir):
    if json_file.endswith(".json"):
        key = json_file[:-5]
        with open(os.path.join(json_dir, json_file), 'r') as f:
            data = json.load(f)
            for shape in data["shapes"]:
                label = shape["label"]

                if is_chinese_only(label) and len(label) > 0 and not '#' in label:
                    points = shape["points"]
                    points_np = np.array(points, dtype=np.float32)

                    if len(label) == 1:
                        # single character
                        if not key in character_boxes:
                            character_boxes[key] = []
                        character_boxes[key].append((points_np, label))
                    else:
                        # big container with multiple characters
                        if not key in metadata:
                            metadata[key] = []
                        metadata[key].append((points_np, label))

def format_pts(pts: np.ndarray):
    result = ""
    for i in range(4):
        result += f"{int(pts[i][0])},{int(pts[i][1])}"
        if i != 3:
            result += ","
    return result

punctuation = "。，、；：「」『』（）─？！…﹏《》〈〉＿．～—"
def filter_punctuation(label: str):
    for i in range(len(punctuation)):
        ch = punctuation[i]
        label = label.replace(ch, "")
    return label
# draw and save
extract_num = re.compile("([0-9]+)")
imgs = sorted([x for x in os.listdir(img_dir) if x.endswith(".jpg")], key=lambda x: int(extract_num.search(x).group(0)))

def remove_from_list(l: list, arr):
    for index, item in enumerate(l):
        if np.array_equal(item[0], arr[0]) and item[1] == arr[1]:
            l.pop(index)
            return

with open(outupt_csv_path, 'w') as f, open(output_ground_csv_path, 'w') as f_ground:
    for img_name in tqdm(imgs):
        # img_name = "img_143.jpg"
        img = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_UNCHANGED)
        key = img_name[:-4]
        should_move = False
        if key in metadata:

            leftover_character_boxes = character_boxes[key].copy()

            for index, (container_pts, container_label) in enumerate(metadata[key]):
                boxes = []

                for character_box in character_boxes[key]:
                    character_box_pts, label = character_box
                    if character_in_container(character_box_pts, container_pts):
                        boxes.append(character_box)
                        remove_from_list(leftover_character_boxes, character_box)


                if len(boxes) > 0:
                    line = f"{key},{format_pts(container_pts)}"
                    f.write(f"{line}\n")
                    f_ground.write(f"{line},{filter_punctuation(container_label)}\n")
                    should_move = True
                else:
                    print("[W] Found a parent container without any child. Skipping this container...")
                    print(key)
                    # cv2.imshow("test", cropped)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    continue

            for index, (single_box_pts, label) in enumerate(leftover_character_boxes):
                line = f"{key},{format_pts(single_box_pts)}"
                f.write(f"{line}\n")
                f_ground.write(f"{line},{filter_punctuation(label)}\n")
                should_move = True
        
        if should_move:
            shutil.copy(os.path.join(img_dir, img_name), os.path.join(output_img_dir, img_name))

