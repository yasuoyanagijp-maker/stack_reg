import numpy as np
import os
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict
import tifffile
from app.core.validation import validate_folder_structure, validate_images_readable
from app.core.image_proc import (
    create_reference_stack_image5,
    optimize_clahe_parameters,
    enlarge_image_4x,
    imread_grayscale,
    apply_clahe,
)
from app.core.registration import (
    calculate_affine_transformations,
    apply_transformations_to_stack,
    average_project_stack,
    compute_alignment_cc,
    refine_affine_feature_based,
    DEFAULT_CONFIDENCE_THRESHOLD,
    REFINE_MIN_IMPROVEMENT,
)


def discover_visits(input_dir: str):
    """
    Auto-detects whether the input_dir is a Patient directory (containing multiple Visits)
    or a Visit directory (containing Layers).
    """
    subfolders = sorted([f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))])
    if not subfolders:
        return []
    
    first_sub = os.path.join(input_dir, subfolders[0])
    has_images = any(f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg')) for f in os.listdir(first_sub))
    if has_images:
        return [input_dir] # It's a Visit dir
    
    # It's a Patient dir - find all valid Visit subdirectories
    visits = []
    for sf in subfolders:
        visit_dir = os.path.join(input_dir, sf)
        visit_subs = [f for f in os.listdir(visit_dir) if os.path.isdir(os.path.join(visit_dir, f))]
        if visit_subs:
            fs = os.path.join(visit_dir, visit_subs[0])
            if any(f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg')) for f in os.listdir(fs)):
                visits.append(visit_dir)
    return visits


@dataclass
class VisitPlan:
    """
    All intermediate artifacts for a single Visit, computed once so the results
    can be reviewed (and individual capture alignments manually corrected) before
    the expensive per-layer warping/averaging is committed.

    - ``ref_stack[i]`` is the pretreated "Image 5" average for capture ``i``.
    - ``matrices[i]`` maps reference (capture 0) coords -> capture ``i`` coords
      (``cv2.WARP_INVERSE_MAP`` convention).
    - ``scores[i]`` is the automatic alignment confidence for capture ``i``.
    """
    visit_dir: str
    visit_name: str
    folder_contents: Dict[str, List[str]]
    sorted_captures: List[str]
    ref_stack: np.ndarray
    matrices: List[np.ndarray]
    scores: List[float] = field(default_factory=list)

    def low_confidence_indices(
        self, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> List[int]:
        """Capture indices (excluding the anchor) whose auto-alignment is suspect."""
        return [
            i for i, s in enumerate(self.scores)
            if i > 0 and s < threshold
        ]

    def load_layer_stack(self, layer_idx: int) -> np.ndarray:
        """
        Load the 4×-enlarged grayscale images for ``layer_idx`` (0-based) across
        every capture. Same spatial size as ``ref_stack`` (Image 5), so affine
        matrices remain interchangeable for display and warping.
        """
        if not self.sorted_captures:
            raise ValueError("VisitPlan has no captures.")
        n_layers = len(self.folder_contents[self.sorted_captures[0]])
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range for {n_layers} layer(s)."
            )
        slices: List[np.ndarray] = []
        for capture_folder in self.sorted_captures:
            filename = self.folder_contents[capture_folder][layer_idx]
            file_path = os.path.join(self.visit_dir, capture_folder, filename)
            img = imread_grayscale(file_path)
            if img is None:
                raise ValueError(f"Failed to read: {file_path}")
            slices.append(enlarge_image_4x(img))
        return np.stack(slices, axis=0)


def _noop_progress(val: float, status: str):
    pass


def _noop_log(msg: str):
    pass


def auto_refine_matrices(
    ref_stack: np.ndarray,
    matrices: List[np.ndarray],
    scores: List[float],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    min_improvement: float = REFINE_MIN_IMPROVEMENT,
    refine_fn: Callable = refine_affine_feature_based,
    log: Optional[Callable[[str], None]] = None,
) -> List[int]:
    """
    For every capture whose confidence is below ``threshold``, run ``refine_fn``
    (feature-based by default) and adopt its matrix only if it beats the current
    score by more than ``min_improvement`` on the same correlation metric.

    ``matrices`` and ``scores`` are updated in place. Returns the list of capture
    indices whose matrices were replaced. This is the automatic attempt made
    before manual corresponding-point correction.
    """
    log = log or _noop_log
    refined: List[int] = []
    for i in range(1, len(matrices)):
        if scores[i] >= threshold:
            continue
        result = refine_fn(ref_stack[0], ref_stack[i])
        if result is None:
            log(f"  [AUTO-REFINE] Capture {i+1}: feature matching found too few points; keeping auto.")
            continue
        cand_matrix, cand_cc = result
        if cand_cc > scores[i] + min_improvement:
            log(
                f"  [AUTO-REFINE] Capture {i+1} improved {scores[i]:.3f} -> {cand_cc:.3f} "
                "(feature-based ORB+RANSAC)."
            )
            matrices[i] = np.asarray(cand_matrix, dtype=np.float32)
            scores[i] = float(cand_cc)
            refined.append(i)
        else:
            log(
                f"  [AUTO-REFINE] Capture {i+1}: feature-based {cand_cc:.3f} did not beat "
                f"auto {scores[i]:.3f}; keeping auto."
            )
    return refined


def prepare_visit(
    visit_dir: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    auto_refine: bool = True,
) -> VisitPlan:
    """
    Phase 1 for a single Visit: validate, build the "Image 5" reference stack and
    compute the automatic per-capture Affine alignment together with a confidence
    score for each capture.

    When ``auto_refine`` is True, any capture whose intensity-based alignment is
    below the confidence threshold gets an automatic second attempt via a
    feature-based (ORB + RANSAC) estimate; the better of the two (measured with
    the same correlation metric) is kept. Manual corresponding-point correction
    is started later by the user from the post-run review screen (visual
    selection of a result image) — not by automatic flagging here.

    ``progress_callback`` receives values in ``0.0..1.0`` scoped to this phase.
    Raises ``ValueError`` on validation / IO failures.
    """
    progress = progress_callback or _noop_progress
    log = log_callback or _noop_log

    visit_name = os.path.basename(visit_dir.rstrip(os.sep))

    progress(0.02, f"[{visit_name}] Validating structure...")
    is_valid, msg, folder_contents = validate_folder_structure(visit_dir)
    if not is_valid:
        raise ValueError(f"{visit_name}: {msg}")
    log(f"  {msg}")

    progress(0.05, f"[{visit_name}] Checking image files...")
    readable, read_msg = validate_images_readable(visit_dir, folder_contents)
    if not readable:
        raise ValueError(f"{visit_name}: {read_msg}")
    log(f"  {read_msg}")

    def image5_cb(inner_val, inner_status):
        progress(0.05 + inner_val * 0.35, f"[{visit_name}] {inner_status}")

    ref_stack = create_reference_stack_image5(
        visit_dir, folder_contents, progress_callback=image5_cb
    )
    log(f"  Image 5 reference stack created. Slices: {ref_stack.shape[0]}")

    def align_cb(inner_val, inner_status):
        progress(0.40 + inner_val * 0.60, f"[{visit_name}] {inner_status}")

    matrices, _ecc_scores = calculate_affine_transformations(
        ref_stack,
        progress_callback=align_cb,
        log_callback=log,
        return_scores=True,
    )

    if not np.allclose(matrices[0], np.eye(2, 3)):
        log("  [WARN] Alignment Anchor (Slice 1) is not Identity. Results may be shifted.")

    # Use a single, consistent correlation metric for every capture so automatic,
    # feature-refined and manual results are all comparable on the same scale.
    scores = [1.0]
    for i in range(1, len(matrices)):
        scores.append(compute_alignment_cc(ref_stack[0], ref_stack[i], matrices[i]))

    if auto_refine:
        auto_refine_matrices(ref_stack, matrices, scores, log=log)

    progress(1.0, f"[{visit_name}] Alignment ready.")

    plan = VisitPlan(
        visit_dir=visit_dir,
        visit_name=visit_name,
        folder_contents=folder_contents,
        sorted_captures=sorted(folder_contents.keys()),
        ref_stack=ref_stack,
        matrices=matrices,
        scores=scores,
    )
    return plan


def finalize_visit(
    plan: VisitPlan,
    patient_name: str,
    patient_output_dir: str,
    apply_clahe_to_ref: bool = False,
    automate_tuning: bool = True,
    matrix_overrides: Optional[Dict[int, np.ndarray]] = None,
    source_layer: Optional[int] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Phase 2 for a single Visit: apply the (optionally manually corrected) capture
    matrices to every layer (still via the 4× enlarge path), enhance with CLAHE,
    average-project and save.

    ``matrix_overrides`` maps a capture index to a replacement 2x3 matrix (as
    produced by ``estimate_affine_from_correspondences``); these take precedence
    over the automatic alignment for the affected captures.

    The same per-capture matrices are applied to **all** layers (image1…imageN).
    When the user corrected landmarks on one selected result image, pass that
    0-based index as ``source_layer`` so the journal records which image the
    parameters came from; the transforms still rewrite every output image.

    ``progress_callback`` receives values in ``0.0..1.0`` scoped to this phase.
    """
    progress = progress_callback or _noop_progress
    log = log_callback or _noop_log

    visit_name = plan.visit_name
    folder_contents = plan.folder_contents
    sorted_captures = plan.sorted_captures

    # Apply manual overrides without mutating the plan's original matrices.
    matrices = [m.copy() for m in plan.matrices]
    if matrix_overrides:
        for idx, mat in matrix_overrides.items():
            if 0 <= idx < len(matrices):
                matrices[idx] = np.asarray(mat, dtype=np.float32)
                log(f"  [MANUAL] Capture {idx+1} using manual corresponding-point matrix.")

    total_captures = len(sorted_captures)
    total_layers = len(folder_contents[sorted_captures[0]])

    if matrix_overrides:
        src = (
            f"image{source_layer + 1}"
            if source_layer is not None
            else "manual corresponding points"
        )
        all_imgs = ", ".join(f"image{i + 1}" for i in range(total_layers))
        log(
            f"  [MANUAL] Parameters from {src} will be applied to all layers "
            f"({all_imgs})."
        )

    for layer_idx in range(total_layers):
        layer_weight = 1.0 / total_layers
        layer_base = layer_idx * layer_weight

        log(f"  Processing Layer {layer_idx+1}/{total_layers}...")
        log(f"    Loading {total_captures} captures (4x enlarge)...")

        raw_stack = []
        for cap_idx, capture_folder in enumerate(sorted_captures):
            image_files = folder_contents[capture_folder]
            filename = image_files[layer_idx]
            file_path = os.path.join(plan.visit_dir, capture_folder, filename)
            log(f"    Capture {cap_idx + 1}/{total_captures}: {filename}")

            img = imread_grayscale(file_path)
            if img is None:
                log(f"ERROR: Failed to read: {file_path}")
                return False

            raw_stack.append(enlarge_image_4x(img))

        stack_array = np.stack(raw_stack, axis=0)
        num_slices = stack_array.shape[0]
        log(f"    Warping {num_slices} slices...")

        def layer_warp_cb(inner_val, inner_status):
            progress(layer_base + (inner_val * 0.5 * layer_weight), f"[{visit_name}] L{layer_idx+1}: {inner_status}")

        # Same matrices for every layer — manual params from one image apply to all.
        registered_stack = apply_transformations_to_stack(stack_array, matrices, progress_callback=layer_warp_cb)
        log(f"    Warp complete.")

        if apply_clahe_to_ref or layer_idx > 0:
            mid_slice = (num_slices + 1) // 2 - 1
            if automate_tuning:
                log(f"    Optimizing CLAHE parameters (grid search)...")
                progress(layer_base + 0.5 * layer_weight, f"[{visit_name}] L{layer_idx+1}: Finding optimal CLAHE...")
                b, h, s = optimize_clahe_parameters(registered_stack[mid_slice])
                log(f"  Layer {layer_idx + 1} CLAHE: BlockSize={b}px, Bins={h}, Slope={s}")
            else:
                b, h, s = (32, 256, 4.0)
                log(f"  Layer {layer_idx + 1} CLAHE (fixed): BlockSize={b}px, Bins={h}, Slope={s}")

            log(f"    Applying CLAHE to {num_slices} slices...")
            for s_idx in range(num_slices):
                clahe_p = layer_base + (0.5 * layer_weight) + ((s_idx / num_slices) * 0.4 * layer_weight)
                progress(clahe_p, f"[{visit_name}] L{layer_idx+1}: Enhancing {s_idx+1}/{num_slices}...")
                registered_stack[s_idx] = apply_clahe(
                    registered_stack[s_idx], clip_limit=s, block_size=b, nbins=h
                )
        elif layer_idx == 0 and not apply_clahe_to_ref:
            log(f"    Skipping CLAHE for Layer 1 (reference layer).")

        progress(layer_base + 0.9 * layer_weight, f"[{visit_name}] L{layer_idx+1}: Averaging...")
        avg_img = average_project_stack(registered_stack)

        output_filename = f"{patient_name}-Avg-Stack_{visit_name}_image{layer_idx+1}.tif"
        output_filepath = os.path.join(patient_output_dir, output_filename)
        tifffile.imwrite(output_filepath, avg_img)
        log(f"    Saved: {output_filename}")

    progress(1.0, f"[{visit_name}] Done.")
    return True


def run_registration_pipeline(
    input_dir: str, 
    output_dir: str, 
    apply_clahe_to_ref: bool = False,
    automate_tuning: bool = True,
    overrides_by_visit: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    auto_refine: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None
) -> Optional[List[VisitPlan]]:
    """
    Orchestrates the entire OCTA Registration process with strict parity 
    to the ImageJ macro logic, supporting both single Visit and full Patient structures.

    ``overrides_by_visit`` optionally maps a visit name to ``{capture_index: matrix}``
    manual corresponding-point corrections, which override the automatic alignment
    for those captures.

    Returns the list of prepared ``VisitPlan`` objects on success (kept for a
    subsequent visual Review & Correct without re-running alignment), or
    ``None`` on failure.
    """
    
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def progress(val: float, status: str):
        if progress_callback:
            progress_callback(val, status)

    log("--- Starting Pipeline ---")
    progress(0.05, "Scanning directory structure...")
    
    visits = discover_visits(input_dir)
    if not visits:
        log("ERROR: No valid Visit folders found. Expected a folder containing visits, or a visit containing layers.")
        return None

    patient_name = os.path.basename(input_dir.rstrip(os.sep))
    patient_output_dir = os.path.join(output_dir, patient_name)
    if not os.path.exists(patient_output_dir):
        os.makedirs(patient_output_dir)
        log(f"Created output subdirectory: {patient_output_dir}")

    total_visits = len(visits)
    overrides_by_visit = overrides_by_visit or {}
    plans: List[VisitPlan] = []

    for v_idx, visit_dir in enumerate(visits):
        visit_name = os.path.basename(visit_dir.rstrip(os.sep))
        log(f"\n--- Processing Visit {v_idx+1}/{total_visits}: {visit_name} ---")

        base_prog = (v_idx / total_visits)
        v_scale = 1.0 / total_visits

        # Phase 1 (prepare) occupies 0..0.6 of the visit budget, phase 2 the rest.
        def prep_progress(val, status):
            progress(base_prog + (val * 0.6) * v_scale, status)

        def final_progress(val, status):
            progress(base_prog + (0.6 + val * 0.4) * v_scale, status)

        try:
            plan = prepare_visit(visit_dir, progress_callback=prep_progress, log_callback=log, auto_refine=auto_refine)
        except ValueError as e:
            log(f"ERROR in {visit_name}: {e}")
            return None
        except Exception as e:
            log(f"ERROR creating reference stack for {visit_name}: {str(e)}")
            return None

        ok = finalize_visit(
            plan,
            patient_name=patient_name,
            patient_output_dir=patient_output_dir,
            apply_clahe_to_ref=apply_clahe_to_ref,
            automate_tuning=automate_tuning,
            matrix_overrides=overrides_by_visit.get(visit_name),
            progress_callback=final_progress,
            log_callback=log,
        )
        if not ok:
            return None
        plans.append(plan)

    progress(1.0, "Completed successfully.")
    log("\n--- Pipeline Finished ---")
    return plans
