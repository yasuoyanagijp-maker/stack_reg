import cv2
import numpy as np
import tifffile
import os
from skimage.metrics import structural_similarity as ssim

def compare_images(p1, p2):
    img1 = tifffile.imread(p1)
    img2 = tifffile.imread(p2)
    
    if img1.shape != img2.shape:
        return f"Size mismatch: {img1.shape} vs {img2.shape}"
    
    # Calculate MSE
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    
    # Calculate SSIM (Structural Similarity)
    # data_range=255 for 8-bit images
    score, diff = ssim(img1, img2, full=True, data_range=255)
    
    return {"mse": mse, "ssim": score}

targets = [
    "Patient2/Patient2-Avg-Stack_Visit1_image1.tif",
    "Patient2/Patient2-Avg-Stack_Visit2_image1.tif"
]

ij_root = "/Users/yy/Desktop/imageJresults"
py_root = "/Users/yy/Desktop/pythonresults"

print(f"{'Image Name':<50} | {'SSIM':<8} | {'MSE':<8}")
print("-" * 75)

for t in targets:
    p1 = os.path.join(ij_root, t)
    p2 = os.path.join(py_root, t)
    
    if not os.path.exists(p1) or not os.path.exists(p2):
        print(f"File missing: {t}")
        continue
        
    res = compare_images(p1, p2)
    if isinstance(res, str):
        print(f"{t:<50} | {res}")
    else:
        print(f"{t:<50} | {res['ssim']:.4f} | {res['mse']:.2f}")

