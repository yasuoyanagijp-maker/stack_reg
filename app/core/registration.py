import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Callable

# Captures whose finest-level alignment correlation falls below this value are
# treated as "suspect" and surfaced for optional manual corresponding-point review.
# ECC MOTION_AFFINE correlation on well-aligned pretreated OCTA averages is
# typically > 0.9; genuinely failed alignments sit well below this.
DEFAULT_CONFIDENCE_THRESHOLD = 0.80

# A feature-based refinement is only adopted over the intensity-based result when
# it improves the (identical) alignment-correlation metric by at least this much.
REFINE_MIN_IMPROVEMENT = 0.02


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
    log_callback: Optional[Callable[[str], None]] = None,
    return_scores: bool = False,
):
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
        return_scores: When True, also return a per-slice confidence score
            (finest-resolution ECC correlation, in roughly 0..1) so callers can
            flag captures whose automatic alignment is unreliable and route them
            to manual corresponding-point review.
        
    Returns:
        A list of 2x3 Affine transformation matrices, or, when ``return_scores``
        is True, a tuple ``(matrices, scores)`` where ``scores[i]`` is the
        confidence for slice ``i`` (slice 0, the anchor, is always 1.0).
    """
    logger = logging.getLogger(__name__)
    num_slices = reference_stack.shape[0]
    
    # Anchor (Slice 0) is always Identity.
    matrices = [np.eye(2, 3, dtype=np.float32)]
    # Confidence per slice; the anchor is a perfect self-match by definition.
    scores: List[float] = [1.0]
    
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
        # Confidence for this slice: the finest-level ECC correlation. Stays low
        # if ECC never succeeds at full resolution (i.e. the alignment failed).
        slice_score = 0.0
        seed_val = 0.0
        
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
                    
                    seed_val = float(max_val)
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

                # Record the finest-resolution correlation as the slice confidence.
                if level_res == 0:
                    slice_score = float(cc)
                
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
        # Fall back to the coarse matchTemplate seed strength when ECC never
        # produced a finest-level correlation (alignment effectively failed).
        scores.append(slice_score if slice_score > 0.0 else seed_val)

        if slice_score < DEFAULT_CONFIDENCE_THRESHOLD:
            warn = (
                f"  [LOW CONFIDENCE] Slice {i+1} alignment score "
                f"{scores[-1]:.3f} (< {DEFAULT_CONFIDENCE_THRESHOLD}). "
                "Consider manual corresponding-point correction."
            )
            logger.warning(warn)
            if log_callback:
                log_callback(warn)

        if progress_callback:
            progress_callback(i / total_alignments, f"Slice {i+1}/{num_slices} Aligned.")
        
        del source_pyramid
        
    del target_pyramid
    
    print("  Alignment calculation complete.\n")
    if return_scores:
        return matrices, scores
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

def estimate_affine_from_correspondences(
    reference_points: np.ndarray,
    source_points: np.ndarray,
) -> np.ndarray:
    """
    Builds a 2x3 Affine matrix from manually-picked corresponding points.

    The returned matrix follows the exact same convention as
    ``calculate_affine_transformations`` / ``findTransformECC``: it maps
    *reference* (anchor / slice-0) coordinates to *source* (the slice being
    aligned) coordinates, so it can be dropped straight into
    ``apply_transformations_to_stack`` (which uses ``cv2.WARP_INVERSE_MAP``).

    In other words, for each landmark the user clicks the same anatomical
    feature in the reference image and in the source image; this estimates the
    transform such that ``M @ [x_ref, y_ref, 1] == [x_src, y_src]``.

    Args:
        reference_points: (N, 2) array of (x, y) points in the reference image.
        source_points:    (N, 2) array of matching (x, y) points in the source image.

    Returns:
        A 2x3 float32 Affine matrix.

    Raises:
        ValueError: if fewer than 3 correspondences are supplied, the point
            counts differ, or a stable transform cannot be estimated.
    """
    ref = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    src = np.asarray(source_points, dtype=np.float32).reshape(-1, 2)

    if ref.shape[0] != src.shape[0]:
        raise ValueError(
            f"Point counts differ: {ref.shape[0]} reference vs {src.shape[0]} source."
        )
    if ref.shape[0] < 3:
        raise ValueError(
            f"At least 3 corresponding points are required, got {ref.shape[0]}."
        )

    if ref.shape[0] == 3:
        # An exact 3-point affine is fully determined.
        matrix = cv2.getAffineTransform(ref, src)
    else:
        # 4+ points: least-squares fit, robust to a mis-clicked pair.
        matrix, _inliers = cv2.estimateAffine2D(
            ref, src, method=cv2.LMEDS, refineIters=50
        )

    if matrix is None:
        raise ValueError(
            "Could not estimate an affine transform from the given points. "
            "Ensure the points are not collinear and pairs are ordered consistently."
        )

    return matrix.astype(np.float32)


def compute_alignment_cc(
    reference: np.ndarray,
    source: np.ndarray,
    matrix: np.ndarray,
) -> float:
    """
    Scores how well ``matrix`` aligns ``source`` onto ``reference`` using the
    zero-mean normalized cross-correlation over the region that maps inside the
    source frame. Higher is better (1.0 == identical). This mirrors the
    confidence produced by the automatic pyramid alignment so manual results can
    be compared on the same scale.
    """
    h, w = reference.shape[:2]
    ref = reference.astype(np.float32)
    src = source.astype(np.float32)

    # Warp source into the reference frame (matrix maps ref->source coords).
    warped = cv2.warpAffine(
        src, matrix.astype(np.float32), (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Only evaluate where the warp actually produced signal, so the black
    # border introduced by out-of-frame regions does not inflate the score.
    mask = warped > 0
    if mask.sum() < 16:
        return 0.0

    a = ref[mask]
    b = warped[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom <= 1e-9:
        return 0.0
    return float(np.clip((a * b).sum() / denom, -1.0, 1.0))


def refine_affine_feature_based(
    reference: np.ndarray,
    source: np.ndarray,
    max_features: int = 2000,
    ratio: float = 0.75,
    ransac_thresh: float = 5.0,
    full_affine: bool = False,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Feature-based (ORB + RANSAC) affine estimate, used as an automatic second
    attempt for captures where the intensity-based pyramid alignment is
    unreliable (typically low-SNR source averages where ``matchTemplate`` seeds
    poorly or ECC settles in a wrong local minimum).

    This is CPU-only and dependency-free (OpenCV), unlike deep-feature methods,
    and it fails in a *different* way than intensity ECC, so it often recovers
    captures with large translation/rotation offsets. By default it fits a
    partial affine (rotation + uniform scale + translation), which preserves
    vascular geometry for same-eye OCTA averaging; set ``full_affine=True`` to
    allow shear/anisotropic scale.

    The returned matrix follows the ``WARP_INVERSE_MAP`` convention used
    throughout this module (reference coords -> source coords), so it is directly
    interchangeable with the automatic matrices and manual corrections.

    Returns:
        ``(matrix, cc)`` where ``cc`` is ``compute_alignment_cc`` of the estimate,
        or ``None`` when too few reliable correspondences are found.
    """
    ref = reference.astype(np.uint8)
    src = source.astype(np.uint8)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp_ref, des_ref = orb.detectAndCompute(ref, None)
    kp_src, des_src = orb.detectAndCompute(src, None)
    if des_ref is None or des_src is None or len(kp_ref) < 3 or len(kp_src) < 3:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_ref, des_src, k=2)
    good = [pair[0] for pair in knn
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    if len(good) < 3:
        return None

    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good])
    src_pts = np.float32([kp_src[m.trainIdx].pt for m in good])

    if full_affine:
        matrix, _ = cv2.estimateAffine2D(
            ref_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
        )
    else:
        matrix, _ = cv2.estimateAffinePartial2D(
            ref_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
        )
    if matrix is None:
        return None

    matrix = matrix.astype(np.float32)
    cc = compute_alignment_cc(reference, source, matrix)
    return matrix, cc


def average_project_stack(stack: np.ndarray) -> np.ndarray:
    """
    Computes the Average Intensity Projection of a stack.
    
    In ImageJ: run("Z Project...", "projection=[Average Intensity]");
    """
    return np.mean(stack, axis=0).astype(np.uint8)
