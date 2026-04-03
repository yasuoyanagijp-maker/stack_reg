import cv2
import numpy as np
import os
import logging
import math
from typing import Optional, List, Dict, Tuple

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

def apply_clahe(image: np.ndarray, clip_limit: float = 3.0, tile_size: int = 127) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Matches ImageJ's "Enhance Local Contrast".
    
    Args:
        image: Grayscale image.
        clip_limit: The maximum slope of the histogram contrast.
        tile_size: The grid size for histogram calculation (blocksize).
        
    Returns:
        Enhanced image.
    """
    # OpenCV uses tileGridSize as (rows, cols)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)

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
    img = apply_clahe(img, clip_limit=3.0, tile_size=127)
    
    # 3. Gaussian Blur (sigma=2)
    # kernel size for sigma=2 is typically (2*sigma*3 + 1) which is ~13x13
    img = cv2.GaussianBlur(img, (13, 13), sigmaX=2)
    
    return img

def create_reference_stack_image5(main_dir: str, folder_contents: Dict[str, List[str]]) -> np.ndarray:
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
        
    Returns:
        A 3D NumPy array representing the 'Image 5' reference stack.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting creation of Image 5 reference stack (Stack of Averages)...")
    
    folder_averages: List[np.ndarray] = []
    
    # Sort folders to ensure consistent stack order
    sorted_folders = sorted(folder_contents.keys())
    
    for folder_name in sorted_folders:
        logger.info(f"  Processing folder: {folder_name}")
        image_files = folder_contents[folder_name]
        
        temp_stack = []
        for filename in image_files:
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
    
    # Parameters to test (from original macro)
    blocksizes = [8, 16, 32]
    hist_bins_array = [128, 256]
    max_slopes = [2.0, 3.0, 4.0]
    
    best_score = -1.0
    best_params = (16, 256, 3.0) # Defaults
    
    logger.info("Starting grid search for optimal CLAHE parameters...")
    
    for b in blocksizes:
        for h in hist_bins_array:
            for s in max_slopes:
                # Apply CLAHE with current parameters
                test_img = apply_clahe(image, clip_limit=s, tile_size=b)
                
                # Calculate quality
                score = calculate_quality_score(test_img)
                
                if score > best_score:
                    best_score = score
                    best_params = (b, h, s)
                    
    logger.info(f"Optimal parameters found: Block={best_params[0]}, Bins={best_params[1]}, "
                f"Slope={best_params[2]} (Score: {best_score:.4f})")
                
    return best_params
