import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

def build_gaussian_pyramid(img: np.ndarray, min_size: int = 64) -> List[np.ndarray]:
    """
    Builds a multi-level Gaussian pyramid by downsampling until traversing below min_size.
    Matches ImageJ TurboReg's coarse-to-fine optimization strategy.
    """
    pyramid = [img]
    while True:
        h, w = pyramid[-1].shape[:2]
        if h <= min_size * 2 or w <= min_size * 2:
            break
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid

def calculate_affine_transformations(reference_stack: np.ndarray) -> List[np.ndarray]:
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
    
    # Store transformation matrices. The first one is identity (no movement).
    matrices = [np.eye(2, 3, dtype=np.float32)]
    
    # The first slice is our absolute reference
    target_slice = reference_stack[0].astype(np.float32)
    target_pyramid = build_gaussian_pyramid(target_slice)
    num_levels = len(target_pyramid)
    
    # ECC algorithm parameters
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
    
    # Scaling matrices for hierarchical translation step-up (M_large = S * M_small * S_inv)
    S = np.array([[2.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    S_inv = np.array([[0.5, 0.0, 0.0],
                      [0.0, 0.5, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
                      
    logger.info(f"Starting Multi-resolution Affine alignment ({num_levels} pyramid levels) on {num_slices} slices...")
    print(f"Starting Multi-resolution Affine alignment ({num_levels} levels) on {num_slices} slices...")
    
    for i in range(1, num_slices):
        source_slice = reference_stack[i].astype(np.float32)
        source_pyramid = build_gaussian_pyramid(source_slice)
        
        # Initialize the warp matrix as identity for the coarsest level
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Traverse pyramid from Coarse (small image) to Fine (large real image)
        for level in reversed(range(num_levels)):
            curr_target = target_pyramid[level]
            curr_source = source_pyramid[level]
            
            try:
                # 1. ECCで縮小画像同士のズレを計算 (Levenberg-Marquardt最適化)
                # target -> source coordinate map
                (cc, warp_matrix) = cv2.findTransformECC(
                    curr_target, curr_source, warp_matrix, 
                    cv2.MOTION_AFFINE, criteria
                )
                logger.info(f"  Slice {i} Level {level} alignment completed (Correlation: {cc:.4f})")
                print(f"  Slice {i} Level {level} alignment completed (Correlation: {cc:.4f})") # Print to console
            except cv2.error as e:
                logger.warning(f"  Slice {i} Level {level} alignment failed: {str(e)}")
                print(f"  Slice {i} Level {level} alignment failed: {str(e)}")
                # 失敗時でも前階層から引き継いだ warp_matrix の初期値を保持して利用する
                pass
            
            # 【重要】高精度なスケーリング処理 (M_large = S * M_small * S^-1)
            # 現在のレベルが0段階(フルサイズ)でなければ、次階層(2倍解像度)にむけて行列をスケールアップ
            if level > 0:
                M_3x3 = np.vstack([warp_matrix.astype(np.float64), [0.0, 0.0, 1.0]])
                M_scaled = S @ M_3x3 @ S_inv
                warp_matrix = M_scaled[:2, :].astype(np.float32)

        matrices.append(warp_matrix)
        
    return matrices

def apply_transformations_to_stack(stack: np.ndarray, matrices: List[np.ndarray]) -> np.ndarray:
    """
    Applies a list of Affine transformation matrices to a stack of images.
    
    Matches Phase 2 of the ImageJ macro logic.
    
    Args:
        stack: A 3D NumPy array (Slices, Height, Width).
        matrices: A list of 2x3 Affine transformation matrices.
        
    Returns:
        The transformed stack.
    """
    num_slices = stack.shape[0]
    h, w = stack.shape[1], stack.shape[2]
    
    transformed_stack = np.zeros_like(stack)
    
    for i in range(num_slices):
        matrix = matrices[i]
        # warpAffine applies the transformation.
        # findTransformECC returns a matrix that maps template(target) -> input(source).
        # We must use cv2.WARP_INVERSE_MAP so warpAffine interprets the matrix as dst->src and 
        # fetches the pixels from the source correctly.
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
    
    Args:
        stack: A 3D NumPy array.
        
    Returns:
        A 2D NumPy array (Average Intensity).
    """
    return np.mean(stack, axis=0).astype(np.uint8)
