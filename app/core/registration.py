import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Callable

def build_gaussian_pyramid(img: np.ndarray, min_size: int = 64) -> List[np.ndarray]:
    """
    Builds a multi-level Gaussian pyramid by downsampling until traversing below min_size.
    Matches ImageJ TurboReg's coarse-to-fine optimization strategy.
    
    Using cv2.pyrDown provides Gaussian smoothing before downsampling, 
    which is essential for anti-aliasing in registration.
    
    Setting min_size=64 (default) results in approximately 4 levels for 512x512 images,
    providing a robust balance between structural capture and detail preservation.
    """
    pyramid = [img]
    while True:
        h, w = pyramid[-1].shape[:2]
        # Stop if the next level would be smaller than our target minimum size
        if h <= min_size * 2 or w <= min_size * 2:
            break
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid

def calculate_affine_transformations(
    reference_stack: np.ndarray, 
    progress_callback: Optional[Callable[[float, str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None
) -> List[np.ndarray]:
    """
    Calculates the Affine transformation matrix for each slice in the 
    reference stack relative to the first slice.
    
    High-Precision Pyramid Logic (for Perfect Peripheral Alignment):
    1.  Initial Seed: `cv2.matchTemplate` at the coarsest level for global position.
    2.  Full Affine Refinement: Use `cv2.MOTION_AFFINE` at ALL pyramid levels for maximum 
        precision across the entire image field, especially at the periphery.
    3.  Tighter Convergence: 200 iterations and 1e-5 epsilon to ensure "perfect" snapping.
    
    Args:
        reference_stack: A 3D NumPy array (Slices, Height, Width).
        
    Returns:
        A list of 2x3 Affine transformation matrices.
    """
    logger = logging.getLogger(__name__)
    num_slices = reference_stack.shape[0]
    
    # Anchor (Slice 0) is always Identity.
    matrices = [np.eye(2, 3, dtype=np.float32)]
    
    target_slice = reference_stack[0].astype(np.float32)
    # Using min_size=64 for a robust pyramid structure.
    target_pyramid = build_gaussian_pyramid(target_slice, min_size=64)
    num_levels = len(target_pyramid)
    
    # Increased precision for medical image alignment: 200 iterations, 1e-5 epsilon.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    
    # Scaling matrices for hierarchical propagation (M_large = S * M_small * S_inv)
    S = np.array([[2.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    S_inv = np.array([[0.5, 0.0, 0.0],
                      [0.0, 0.5, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
                      
    logger.info(f"Starting Precision Multi-resolution Registration ({num_levels} levels) on {num_slices} slices...")
    
    total_alignments = num_slices - 1
    for i in range(1, num_slices):
        source_slice = reference_stack[i].astype(np.float32)
        source_pyramid = build_gaussian_pyramid(source_slice, min_size=64)
        
        effective_levels = min(num_levels, len(source_pyramid))
        
        # Initial guess (Identity)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Pyramid traversal: Coarse (small image) -> Fine (large image)
        for level_idx, level_res in enumerate(reversed(range(effective_levels))):
            curr_target = target_pyramid[level_res]
            curr_source = source_pyramid[level_res]
            
            msg = f"Aligning Slice {i+1}/{num_slices} [Level {level_res}]"
            if progress_callback:
                level_prog = ((i - 1) / total_alignments) + (level_idx / (effective_levels * total_alignments))
                progress_callback(level_prog, msg)

            try:
                # --- Step A: INITIAL SEED (Only at the coarsest level) ---
                if level_idx == 0:
                    # use matchTemplate to find the best integer translation initial guess
                    res = cv2.matchTemplate(curr_source, curr_target, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    
                    if max_val > 0.4:
                        # Translate current source by max_loc to align with target
                        warp_matrix[0, 2] = float(max_loc[0])
                        warp_matrix[1, 2] = float(max_loc[1])

                # --- Step B: PRECISION AFFINE REFINEMENT ---
                # MOTION_AFFINE is used throughout to ensure edges align as well as the center.
                (cc, new_warp) = cv2.findTransformECC(
                    curr_target, curr_source, warp_matrix, 
                    cv2.MOTION_AFFINE, criteria
                )
                warp_matrix = new_warp
                
                if level_res == 0 or level_res == effective_levels - 1:
                    status_line = f"  Slice {i} Lev {level_res} CC: {cc:.4f} (Precision Affine)"
                    logger.info(status_line)
                    if log_callback:
                        log_callback(status_line)
                
            except cv2.error:
                # If ECC fails at a level, stick with the estimate from the previous level.
                err_msg = f"  [WARN] Slice {i} Lev {level_res} failed. Keeping previous estimate."
                logger.warning(err_msg)
                if log_callback:
                    log_callback(err_msg)
                pass
            
            # Step-up scaling for next resolution level
            if level_res > 0:
                M_3x3 = np.vstack([warp_matrix.astype(np.float64), [0.0, 0.0, 1.0]])
                M_scaled = S @ M_3x3 @ S_inv
                warp_matrix = M_scaled[:2, :].astype(np.float32)
        
        matrices.append(warp_matrix)
        
        if progress_callback:
            progress_callback(i / total_alignments, f"Slice {i+1}/{num_slices} Aligned.")
        
        del source_pyramid
        
    del target_pyramid
    
    print("  Alignment calculation complete.\n")
    return matrices

def apply_transformations_to_stack(
    stack: np.ndarray, 
    matrices: List[np.ndarray],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> np.ndarray:
    """
    Applies a list of Affine transformation matrices to a stack of images.
    
    Matches Phase 2 of the ImageJ macro logic.
    
    Args:
        stack: A 3D NumPy array (Slices, Height, Width).
        matrices: A list of 2x3 Affine transformation matrices.
        progress_callback: Optional progress reporter for UI.
        
    Returns:
        The transformed stack.
    """
    num_slices = stack.shape[0]
    h, w = stack.shape[1], stack.shape[2]
    
    transformed_stack = np.zeros_like(stack)
    
    for i in range(num_slices):
        if progress_callback:
            progress_callback(i / num_slices, f"Warping Slice {i+1}/{num_slices}...")

        matrix = matrices[i]
        # findTransformECC returns a matrix that maps template(target) -> input(source).
        # We must use cv2.WARP_INVERSE_MAP so warpAffine interprets the matrix as dst->src.
        transformed_stack[i] = cv2.warpAffine(
            stack[i], matrix, (w, h), 
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
    return transformed_stack

def average_project_stack(stack: np.ndarray) -> np.ndarray:
    """
    Computes the Average Intensity Projection of a stack.
    
    In ImageJ: run("Z Project...", "projection=[Average Intensity]");
    """
    return np.mean(stack, axis=0).astype(np.uint8)
