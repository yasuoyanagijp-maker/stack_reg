import cv2
import numpy as np

src = np.zeros((10, 10), dtype=np.float32)
src[0, 5] = 255

template = np.zeros((10, 10), dtype=np.float32)
template[0, 0] = 255

warp_matrix = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-10)
cc, warp_matrix = cv2.findTransformECC(template, src, warp_matrix, cv2.MOTION_TRANSLATION, criteria)

print("ECC Matrix:\n", warp_matrix)

dst_forward = cv2.warpAffine(src, warp_matrix, (10, 10))
print("Forward:", np.unravel_index(np.argmax(dst_forward), dst_forward.shape))

dst_inverse = cv2.warpAffine(src, warp_matrix, (10, 10), flags=cv2.WARP_INVERSE_MAP)
print("Inverse:", np.unravel_index(np.argmax(dst_inverse), dst_inverse.shape))
