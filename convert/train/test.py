import numpy as np
import cv2

def character_in_container(character, container, thresh = 0.7):

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
        print(mask.shape)
        points_norm = points - np.array([min_x, min_y])
        print(points_norm)
        cv2.fillPoly(mask, [points_norm], 255)
        

    # draw container mask
    draw_points_to_mask(container, container_mask)

    # draw character mask
    draw_points_to_mask(character, character_mask)

    container_mask_binary = container_mask > 0
    character_mask_binary = character_mask > 0

    return np.sum(container_mask_binary & character_mask_binary) / np.sum(character_mask_binary) > thresh

if __name__ == "__main__":
    
    container_pts = np.array([[805, 617], [1052, 613], [1059, 662], [809, 665]])
    # character_pts = np.array([[806, 620], [841, 620], [851, 666], [810, 664]])
    character_pts = np.array([[694, 976], [707, 974], [708, 991], [694, 993]])

    print(character_in_container(character_pts, container_pts))