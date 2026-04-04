import cv2
import numpy as np

tgt = np.zeros((10, 10), dtype=np.float32)
src = np.zeros((10, 10), dtype=np.float32)

warp_matrix = np.full((2, 3), 9.9, dtype=np.float32)

try:
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-10)
    (cc, warp_matrix) = cv2.findTransformECC(tgt, src, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
except Exception as e:
    pass

print(warp_matrix)
