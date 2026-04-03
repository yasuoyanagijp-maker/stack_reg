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
    
    This rigidly adheres to the 'Phase 1: Align' step of ImageJ's MultiStackReg
    using Marquardt-Levenberg multi-resolution Gaussian Pyramids.
    
    Args:
        reference_stack: A 3D NumPy array (Slices, Height, Width) representing Image 5.
        
    Returns:
        A list of 2x3 Affine transformation matrices.
    """
    logger = logging.getLogger(__name__)
    num_slices = reference_stack.shape[0]
    
    # [IMPORTANT] Ensure matrices list starts with an Identity matrix for Slice 0 (Anchor).
    # This ensures Stack_VisitX_image1..4 all match Capture 1 as the reference point.
    matrices = [np.eye(2, 3, dtype=np.float32)]
    
    # The first slice is our absolute reference (Anchor)
    target_slice = reference_stack[0].astype(np.float32)
    target_pyramid = build_gaussian_pyramid(target_slice, min_size=64)
    num_levels = len(target_pyramid)
    
    # ECC algorithm parameters (Levenberg-Marquardt style)
    # 50 iterations and 0.001 epsilon for convergence.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
    
    # Scaling matrices for hierarchical translation step-up (M_large = S * M_small * S_inv)
    S = np.array([[2.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    S_inv = np.array([[0.5, 0.0, 0.0],
                      [0.0, 0.5, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
                      
    logger.info(f"Starting Multi-resolution Affine alignment ({num_levels} pyramid levels) on {num_slices} slices...")
    print(f"\n[Alignment] Phase 1: Calculating transformation parameters ({num_levels} levels)...")
    
    total_alignments = num_slices - 1
    for i in range(1, num_slices):
        source_slice = reference_stack[i].astype(np.float32)
        source_pyramid = build_gaussian_pyramid(source_slice, min_size=64)
        
        # Ensure pyramids have the same depth
        effective_levels = min(num_levels, len(source_pyramid))
        
        # Initialize the warp matrix as identity for the coarsest level
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Traverse pyramid from Coarse (small image) to Fine (large real image)
        for level_idx, level in enumerate(reversed(range(effective_levels))):
            # Calculate granular progress for this level
            msg = f"Aligning Slice {i+1}/{num_slices} (Level {level})..."
            if progress_callback:
                level_prog = ((i - 1) / total_alignments) + (level_idx / (effective_levels * total_alignments))
                progress_callback(level_prog, msg)
            
            # Send message to Journal in UI
            if log_callback:
                log_callback(f"  {msg}")

            curr_target = target_pyramid[level]
            curr_source = source_pyramid[level]
            
            try:
                # 1. ECCで縮小画像同士のズレを計算 (Target -> Source coordinate mapping)
                (cc, warp_matrix) = cv2.findTransformECC(
                    curr_target, curr_source, warp_matrix, 
                    cv2.MOTION_AFFINE, criteria
                )
                logger.info(f"  Slice {i} Level {level} alignment (Correlation: {cc:.4f})")
                
                # Consistently log to the console at key levels
                if level == 0 or level == effective_levels - 1:
                    status_line = f"  Slice {i}/{num_slices-1} Level {level} (Res: {curr_target.shape[1]}x{curr_target.shape[0]}, Corr: {cc:.4f})"
                    print(status_line)
                    if log_callback:
                        log_callback(status_line)
                
            except cv2.error as e:
                err_msg = f"  [WARN] Slice {i} Level {level} did not converge. Using coarse estimate."
                logger.warning(err_msg)
                print(err_msg)
                if log_callback:
                    log_callback(err_msg)
                # Keep the warp_matrix from the previous (coarser) level as the best estimate
                pass
            
            # 【重要】高精度なスケーリング処理 (M_large = S * M_small * S^-1)
            # Propagate the transformation to the next higher resolution level
            if level > 0:
                M_3x3 = np.vstack([warp_matrix.astype(np.float64), [0.0, 0.0, 1.0]])
                M_scaled = S @ M_3x3 @ S_inv
                warp_matrix = M_scaled[:2, :].astype(np.float32)
        
        matrices.append(warp_matrix)
        
        # Final progress for this fully completed slice
        done_msg = f"Slice {i+1}/{num_slices} Aligned."
        if progress_callback:
            progress_callback(i / total_alignments, done_msg)
        if log_callback:
            log_callback(f"  [OK] {done_msg}")
        
        # Memory Management: Clear the source pyramid before next slice
        del source_pyramid
        
    # Memory Management: Clear the target pyramid
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
