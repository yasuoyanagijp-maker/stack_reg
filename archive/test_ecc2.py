import cv2
import numpy as np

tgt = np.zeros((100, 100), dtype=np.float32)
tgt[40:60, 40:60] = 255.0

src = np.zeros((100, 100), dtype=np.float32)
src[40:60, 50:70] = 255.0

warp_matrix = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-10)
_, warp_matrix = cv2.findTransformECC(tgt, src, warp_matrix, cv2.MOTION_TRANSLATION, criteria)

print("ECC matrix: tx =", warp_matrix[0, 2])

dst_no_flag = cv2.warpAffine(src, warp_matrix, (100, 100))
print("Without flag: start col =", np.where(dst_no_flag[40, :] > 0)[0][0])

dst_with_flag = cv2.warpAffine(src, warp_matrix, (100, 100), flags=cv2.WARP_INVERSE_MAP)
print("With flag: start col =", np.where(dst_with_flag[40, :] > 0)[0][0])
