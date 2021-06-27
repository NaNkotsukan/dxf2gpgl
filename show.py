import cv2
import numpy as np

def save(fname, path, size=None):
    if size is None:
        y, x = path.max(axis=0)+2
    else:
        y, x = size+2
    path = path.reshape(-1, 2)
    img = np.full((x, y, 3), 128, dtype=np.uint8)
    COLOR = [(255,0,0), (0,255,0)]
    prev = path[0]
    for i, next in enumerate(path):
        cv2.line(img, tuple(prev), tuple(next), COLOR[i%2], thickness=1, lineType=cv2.LINE_4)
        prev = next
    cv2.circle(img, tuple(prev), 5, (0,0,255))
    cv2.imwrite(f"{fname}.png", img)
