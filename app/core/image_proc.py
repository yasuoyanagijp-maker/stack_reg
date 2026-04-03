import cv2
import numpy as np
import os
import logging
import math
from typing import Optional, List, Dict, Tuple, Callable

def enlarge_image_4x(image: np.ndarray) -> np.ndarray:
    """
    Enlarges the image to 4 times its original size using Bicubic interpolation.
    
    In ImageJ, this matches:
    run("Size...", "width=[w*4] height=[h*4] average interpolation=Bicubic");
    
    Args:
        image: The input image as a NumPy array (grayscale).
        
    Returns:
        The enlarged image.
    """
    h, w = image.shape[:2]
    # In OpenCV, cv2.INTER_CUBIC is equivalent to Bicubic interpolation
    return cv2.resize(image, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

def subtract_background_rolling_ball(image: np.ndarray, radius: int = 50) -> np.ndarray:
    """
    Simulates the "Rolling Ball" background subtraction from ImageJ.
    
    In many cases, a Morphological Top-Hat Operation is the closest equivalent
    in standard OpenCV.
    
    Args:
        image: The input grayscale image.
        radius: The radius of the ball (structuring element).
        
    Returns:
        Image after background subtraction.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    # Top-hat = original - opening (morphology opening removes bright structures smaller than kernel)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 3.0,
    block_size: int = 127,
    nbins: int = 256,
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Matches ImageJ's "Enhance Local Contrast".

    OpenCV uses a tile grid; ImageJ blocksize is the target local window size in pixels.
    Using ceil keeps tile edges closer to block_size than floor (especially when
    width/height are not multiples of block_size).

    For nbins=128, ImageJ builds a 128-bin histogram; we approximate by halving
    levels before CLAHE and doubling after (256-bin CLAHE on compressed data).
    """
    h, w = image.shape[:2]

    num_tiles_x = max(1, math.ceil(w / block_size))
    num_tiles_y = max(1, math.ceil(h / block_size))

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(num_tiles_x, num_tiles_y))
    if nbins == 256:
        return clahe.apply(image)
    if nbins == 128:
        comp = (image.astype(np.uint16) >> 1).astype(np.uint8)
        out = clahe.apply(comp)
        return np.clip(out.astype(np.uint16) << 1, 0, 255).astype(np.uint8)
    raise ValueError(f"nbins must be 128 or 256, got {nbins}")

def pretreat(image: np.ndarray) -> np.ndarray:
    """
    Performs preprocessing on the image exactly like the ImageJ `pretreat()` function.
    
    ImageJ Logic:
    1. Subtract background (rolling ball = 50)
    2. Enhance contrast (CLAHE, block=127, slope=3)
    3. Gaussian Blur (sigma=2)
    
    Args:
        image: The input grayscale image.
        
    Returns:
        The pretreated image.
    """
    # Ensure 8-bit
    if image.dtype != np.uint8:
        # Simple normalization to 8-bit if it's 16-bit or float
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    # 1. Background Subtraction
    img = subtract_background_rolling_ball(image, radius=50)
    
    # 2. CLAHE (Local Contrast Enhancement)
    # ImageJ standard blocksize is 127 pixels for pretreatment.
    img = apply_clahe(img, clip_limit=3.0, block_size=127, nbins=256)
    
    # 3. Gaussian Blur (sigma=2)
    # kernel size for sigma=2 is typically (2*sigma*3 + 1) which is ~13x13
    img = cv2.GaussianBlur(img, (13, 13), sigmaX=2)
    
    return img

def create_reference_stack_image5(
    main_dir: str, 
    folder_contents: Dict[str, List[str]],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> np.ndarray:
    """
    Creates the 'Image 5' reference stack as described in the original ImageJ macro.
    
    The logic is:
    1. For each sub-folder (visit/layer), process its images:
       - Load, 4x Enlarge, and Pre-treat each image.
       - Combine them into a temporary stack.
       - Calculate the 'Average Intensity Projection' (mean across the stack).
    2. Collect all averages (one per folder) into a final 3D NumPy array.
    
    Args:
        main_dir: The path containing the sub-folders.
        folder_contents: Dictionary of folder names to list of image filenames.
        progress_callback: Optional progress reporter for UI.
        
    Returns:
        A 3D NumPy array representing the 'Image 5' reference stack.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting creation of Image 5 reference stack (Stack of Averages)...")
    
    folder_averages: List[np.ndarray] = []
    
    # Sort folders to ensure consistent stack order
    sorted_folders = sorted(folder_contents.keys())
    total_folders = len(sorted_folders)
    
    for f_idx, folder_name in enumerate(sorted_folders):
        logger.info(f"  Processing folder: {folder_name}")
        image_files = folder_contents[folder_name]
        total_files = len(image_files)
        
        temp_stack = []
        for i_idx, filename in enumerate(image_files):
            # Granular progress within Step 1 (Folders)
            if progress_callback:
                # Sub-progress within the current folder
                folder_prog = f_idx / total_folders
                inner_prog = (i_idx / total_files) / total_folders
                progress_callback(folder_prog + inner_prog, f"Building Image 5: {folder_name} ({i_idx+1}/{total_files})...")

            file_path = os.path.join(main_dir, folder_name, filename)
            
            # Read image as grayscale
            # ImageJ's 8-bit conversion is matched here
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"Failed to read image: {file_path}")
                continue
            
            # 1. 4x Enlargement
            img_resized = enlarge_image_4x(img)
            
            # 2. Pre-treatment (Background subtraction, CLAHE, Gaussian Blur)
            img_pretreated = pretreat(img_resized)
            
            temp_stack.append(img_pretreated)
        
        if not temp_stack:
            logger.warning(f"No valid images processed in folder: {folder_name}")
            continue
            
        # Convert list to 3D array (stack)
        stack_array = np.stack(temp_stack, axis=0)
        
        # 3. Z-Project (Average Intensity)
        # Axis 0 is the stack axis
        # In ImageJ: run("Z Project...", "projection=[Average Intensity]");
        avg_img = np.mean(stack_array, axis=0).astype(np.uint8)
        
        folder_averages.append(avg_img)
        logger.info(f"  Successfully averaged {len(temp_stack)} images from {folder_name}.")
        
    if not folder_averages:
        raise ValueError("No folders were successfully processed for the reference stack.")
        
    # Combine all averages into the final 'Image 5' stack
    image5_stack = np.stack(folder_averages, axis=0)
    logger.info(f"Finished creating Image 5 reference stack. Total slices: {len(folder_averages)}")
    
    return image5_stack

def calculate_quality_score(image: np.ndarray) -> float:
    """
    Calculates a quality score for an image based on Entropy and Contrast.
    
    In ImageJ:
    final_score = entropy * 0.8 + contrast * 0.2;
    
    Args:
        image: Grayscale image.
        
    Returns:
        The calculated quality score.
    """
    # 1. Entropy calculation
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    entropy = -np.sum([p * math.log2(p) for p in hist_norm if p > 0])
    
    # 2. Contrast calculation (Normalized Standard Deviation)
    _, std_dev = cv2.meanStdDev(image)
    contrast = std_dev[0][0] / 128.0
    
    return float(entropy * 0.8 + contrast * 0.2)

def optimize_clahe_parameters(image: np.ndarray) -> Tuple[int, int, float]:
    """
    Finds the optimal CLAHE parameters using a grid search, 
    matching ImageJ's `applyOptimalCLAHEToStack` logic.
    
    Args:
        image: The reference image (usually the middle slice of a stack).
        
    Returns:
        A tuple of (best_block_size, best_bins, best_slope).
    """
    logger = logging.getLogger(__name__)
    
    # Parameters to test (from ImageJ macro lines 409-411)
    # blocksize is in PIXELS
    blocksizes = [8, 16, 32]
    hist_bins_array = [128, 256]
    max_slopes = [2.0, 3.0, 4.0]

    best_score = -1.0
    # ImageJ Defaults (lines 414-416)
    best_params = (16, 256, 3.0)

    logger.info(
        "Finding optimal CLAHE parameters (ImageJ parity: blocks %s, bins %s, slopes %s)...",
        blocksizes,
        hist_bins_array,
        max_slopes,
    )

    for b in blocksizes:
        for h in hist_bins_array:
            for s in max_slopes:
                test_img = apply_clahe(image, clip_limit=s, block_size=b, nbins=h)
                
                # Calculate quality
                score = calculate_quality_score(test_img)
                
                if score > best_score:
                    best_score = score
                    best_params = (b, h, s)
                    
    logger.info(f"Optimal parameters found: BlockSize={best_params[0]}px, Bins={best_params[1]}, "
                f"Slope={best_params[2]} (Score: {best_score:.4f})")
                
    return best_params
