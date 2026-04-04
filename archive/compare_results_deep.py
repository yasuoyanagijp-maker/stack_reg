import cv2
import numpy as np
import tifffile
import os
from skimage.metrics import structural_similarity as ssim

def analyze_diff(p1, p2, name):
    img1 = tifffile.imread(p1).astype(np.float32)
    img2 = tifffile.imread(p2).astype(np.float32)
    
    # 1. Intensity diff
    mean_diff = np.mean(img1 - img2)
    std_diff = np.std(img1 - img2)
    
    # 2. Check for shift using phase correlation
    # Convert to float32 for FFT
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    cross_power_spectrum = (f1 * f2.conj()) / (np.abs(f1 * f2.conj()) + 1e-10)
    r = np.fft.ifft2(cross_power_spectrum)
    peak = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    
    # Shift is (peak_y, peak_x). We need to handle wrap-around.
    sy = peak[0] if peak[0] < img1.shape[0]//2 else peak[0] - img1.shape[0]
    sx = peak[1] if peak[1] < img1.shape[1]//2 else peak[1] - img1.shape[1]
    
    # 3. SSIM
    score = ssim(img1.astype(np.uint8), img2.astype(np.uint8), data_range=255)
    
    # 4. Save diff image for model's view_file (thumb)
    diff_vis = cv2.absdiff(img1.astype(np.uint8), img2.astype(np.uint8))
    # Enhance diff for visibility
    diff_vis = cv2.normalize(diff_vis, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(f"diff_{name}.jpg", diff_vis)
    
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "shift": (sx, sy),
        "ssim": score
    }

ij_root = "/Users/yy/Desktop/imageJresults"
py_root = "/Users/yy/Desktop/pythonresults"
target = "Patient2/Patient2-Avg-Stack_Visit1_image1.tif"

p1 = os.path.join(ij_root, target)
p2 = os.path.join(py_root, target)

res = analyze_diff(p1, p2, "v1_i1")
print(f"Analysis for {target}:")
print(f"  Mean Intensity Diff: {res['mean_diff']:.2f}")
print(f"  Std Intensity Diff:  {res['std_diff']:.2f}")
print(f"  Detected Shift:     {res['shift']} pixels")
print(f"  SSIM Score:         {res['ssim']:.4f}")
