import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

def calculate_affine_transformations(reference_stack: np.ndarray) -> List[np.ndarray]:
    """
    Calculates the Affine transformation matrix for each slice in the 
    reference stack relative to the first slice.
    
    This matches the 'Phase 1: Align' step of ImageJ's MultiStackReg.
    
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
    
    # ECC algorithm parameters
    # Matches typical MultiStackReg behavior for high-precision alignment
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
    
    logger.info(f"Starting Affine alignment calculation on {num_slices} slices...")
    
    for i in range(1, num_slices):
        source_slice = reference_stack[i].astype(np.float32)
        
        # Initialize the warp matrix as identity
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        try:
            # findTransformECC finds the warp matrix that maximizes the correlation
            # between the target and source images.
            (cc, warp_matrix) = cv2.findTransformECC(
                target_slice, source_slice, warp_matrix, 
                cv2.MOTION_AFFINE, criteria
            )
            logger.info(f"  Slice {i} alignment completed (Correlation: {cc:.4f})")
        except cv2.error as e:
            logger.warning(f"  Slice {i} alignment failed: {str(e)}. Using Identity matrix.")
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            
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
        # warpAffine applies the transformation
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
