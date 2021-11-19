from genericpath import exists
import os
import json

from numpy.lib.polynomial import poly
from utils.is_chinese_only import is_chinese_only
import numpy as np
import cv2
from tqdm import tqdm
from utils import four_point_transform, ensure_dir, is_chinese_only
from config import train_path, train_containers_path, train_yolo_data_path

# paths
data_dir = train_path
# data_dir = "/home/tingyu/small_train"

img_dir = os.path.join(data_dir, "img")
json_dir = os.path.join(data_dir, "json")

out_dir = train_containers_path
yolo_out_dir = train_yolo_data_path
csv_out_dir = yolo_out_dir

out_dirs = [out_dir, yolo_out_dir, csv_out_dir]
for dir in out_dirs:
    ensure_dir(dir)

class_id = 0
min_side = 25

write_yolo = True
write_drawn = True


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
                        character_boxes[key].append((label, points_np))
                    else:
                        # big container with multiple characters
                        if not key in metadata:
                            metadata[key] = []
                        metadata[key].append((label, points_np))

# draw and save
imgs = [x for x in os.listdir(img_dir) if x.endswith(".jpg")]


def pad_and_save(cropped: np.ndarray, polys: np.ndarray, poly_labels: str, filename_without_ext: str):
    h, w, c = cropped.shape
    side = max(h, w)
    short_side = min(h, w)
    if short_side < min_side:
        return

    padded_img = np.zeros((side, side, c), dtype=np.uint8)

    if w > h:
        y_begin = (side - h) // 2
        y_end = y_begin + h
        polys[:, :, 1] += y_begin
        padded_img[y_begin:y_end, :, :] = cropped
    else:
        x_begin = (side - w) // 2
        x_end = x_begin + w
        polys[:, :, 0] += x_begin
        padded_img[:, x_begin:x_end, :] = cropped

    padded_img_drawn = padded_img.copy()

    # convert to rectangles
    rects = polys_to_rects(polys)

    with open(os.path.join(csv_out_dir, filename_without_ext + ".txt"), 'w') as f:
        for index, (x, y, w, h) in enumerate(rects):

            label = "" 
            try:
                label = poly_labels[index]
            except IndexError:
                pass

            # get center x and y of bonding box
            cx = x + w // 2
            cy = y + h // 2

            # normalize to 0~1
            cx_n = cx / side
            cy_n = cy / side
            w_n = w / side
            h_n = h / side

            if label in ('一', '二', '三'):
                # make width and height the same
                if w_n > h_n:
                    h_n = w_n
                else:
                    w_n = h_n

            cv2.rectangle(padded_img_drawn, 
                (
                    int((cx_n - w_n/2) * side),
                    int((cy_n - h_n/2) * side)
                ),
                (
                    int((cx_n + w_n/2) * side),
                    int((cy_n + h_n/2) * side)
                ), \
                (0, 255, 0), thickness=2)
            f.write(f"{class_id} {cx_n} {cy_n} {w_n} {h_n}\n")
    if write_yolo:
        cv2.imwrite(os.path.join(
            yolo_out_dir, filename_without_ext + ".jpg"), padded_img)

    if write_drawn:
        cv2.imwrite(os.path.join(
            out_dir, filename_without_ext + ".jpg"), padded_img_drawn)


def remove_arr_from_list(l: list, arr: np.ndarray):
    for index, (_, item) in enumerate(l):
        if np.array_equal(item, arr):
            l.pop(index)
            return


for img_name in tqdm(imgs):
    # img_name = "img_143.jpg"
    img = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_UNCHANGED)
    key = img_name[:-4]
    if key in metadata:

        leftover_character_boxes = character_boxes[key].copy()

        for index, (container_label, container) in enumerate(metadata[key]):
            boxes = []

            for character_label, character_box in character_boxes[key]:
                if character_in_container(character_box, container):
                    boxes.append(character_box)
                    remove_arr_from_list(
                        leftover_character_boxes, character_box)

            cropped, M = four_point_transform(img, container)
            cropped_h, cropped_w, _ = cropped.shape

            if len(boxes) > 0:
                boxes_np = np.array(boxes)  # (n_boxes, 4, 2)
                n_boxes, n_points, _ = boxes_np.shape
                boxes_np = np.concatenate(
                    (boxes_np, np.ones((n_boxes, n_points, 1))), axis=2)

                boxes_np = boxes_np.reshape(-1, 3)
                result = np.matmul(M, boxes_np.transpose())
                divided = (result[:2, :] / result[[2], :]).transpose()
                divided = divided.reshape(-1, 4, 2)
            else:
                print("[W] Found a parent container without any child. Skipping this container...")
                print(key)
                cv2.imshow("test", cropped)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                continue

            img_save_name = f"{key}_{index}"
            pad_and_save(cropped, divided, container_label, img_save_name)

        for index, (single_box_label, single_box) in enumerate(leftover_character_boxes):

            cropped, M = four_point_transform(img, single_box)
            cropped_h, cropped_w, _ = cropped.shape

            polys = np.array([
                [
                    [0, 0],
                    [0, cropped_h],
                    [cropped_w, 0],
                    [cropped_w, cropped_h]

                ]
            ], dtype=np.float32)
            img_save_name = f"{key}_s_{index}"
            pad_and_save(cropped, polys, single_box_label, img_save_name)
