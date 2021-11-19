import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_direction(img: np.ndarray):
    img = img[:,:, 0]
    # H x W
    # map = [img == 0]
    horizontal = np.max(img, axis=0)
    horizontal_binary = horizontal > 0
    left = np.argmax(horizontal_binary)

    vertical = np.max(img, axis=1)
    vertical_binary = vertical > 0
    top = np.argmax(vertical_binary)

    print(f"left: {left}")
    print(f"top: {top}")

    if left > 0 and top == 0:
        return "left_right"
    elif left == 0 and top > 0:
        return "top_bottom"
    else:
        return "unknown"

    # plt.plot(range(horizontal.shape[0]), horizontal)
    # plt.plot(vertical, range(vertical.shape[0]))
    # plt.show()

img = cv2.imread("/home/tingyu/projects/competition/utils/generated/yolo/img_6181.jpg_9.jpg", cv2.IMREAD_UNCHANGED)

print(find_direction(img))