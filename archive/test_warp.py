import cv2
import numpy as np

src = np.zeros((10, 10), dtype=np.uint8)
src[0, 0] = 255

M = np.float32([[1, 0, 5], [0, 1, 0]]) # Expected behavior: shift x by 5
dst_forward = cv2.warpAffine(src, M, (10, 10))
print("Forward (no flag):", list(zip(*np.where(dst_forward > 0))))

dst_inverse = cv2.warpAffine(src, M, (10, 10), flags=cv2.WARP_INVERSE_MAP)
print("Inverse (with flag):", list(zip(*np.where(dst_inverse > 0))))
