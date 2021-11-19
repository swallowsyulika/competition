import cv2
import numpy as np

def square_pad(input_img: np.ndarray, mode = cv2.BORDER_REPLICATE, fill = 0):

    h, w, c = input_img.shape

    side = max(w, h)

    if w > h:
        y_begin = int((side - h) / 2)
        padded_img = cv2.copyMakeBorder(input_img, y_begin, y_begin, 0, 0, mode, value=fill)
        # padded_img[y_begin: y_end, :, :] = result

    else:
        x_begin = int((side - w) / 2)
        padded_img = cv2.copyMakeBorder(input_img, 0, 0, x_begin, x_begin, mode, value=fill)
        # padded_img[:, x_begin:x_end, :] = result
    
    return padded_img