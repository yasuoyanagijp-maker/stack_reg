import cv2
import numpy as np
import os
import logging
import tifffile
from typing import Callable, Optional
from app.core.validation import validate_folder_structure
from app.core.image_proc import (
    create_reference_stack_image5, 
    optimize_clahe_parameters, 
    enlarge_image_4x, 
    pretreat, 
    apply_clahe
)
from app.core.registration import (
    calculate_affine_transformations, 
    apply_transformations_to_stack, 
    average_project_stack
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

def run_registration_pipeline(
    input_dir: str, 
    output_dir: str, 
    apply_clahe_to_ref: bool = False,
    automate_tuning: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None
):
    """
    Orchestrates the entire OCTA Registration process with strict parity 
    to the ImageJ macro logic, supporting both single Visit and full Patient structures.
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
        return False

    patient_name = os.path.basename(input_dir.rstrip(os.sep))
    patient_output_dir = os.path.join(output_dir, patient_name)
    if not os.path.exists(patient_output_dir):
        os.makedirs(patient_output_dir)
        log(f"Created output subdirectory: {patient_output_dir}")

    total_visits = len(visits)
    
    for v_idx, visit_dir in enumerate(visits):
        visit_name = os.path.basename(visit_dir.rstrip(os.sep))
        log(f"\n--- Processing Visit {v_idx+1}/{total_visits}: {visit_name} ---")
        
        base_prog = (v_idx / total_visits)
        v_scale = 1.0 / total_visits
        
        # 1. Validation for this Visit
        progress(base_prog + (0.05 * v_scale), f"[{visit_name}] Validating structure...")
        is_valid, msg, folder_contents = validate_folder_structure(visit_dir)
        if not is_valid:
            log(f"ERROR in {visit_name}: {msg}")
            return False
        log(f"  {msg}")

        # 2. Step 1: Create 'Image 5' Reference Stack (Stack of Averages)
        def image5_progress_cb(inner_val, inner_status):
            # Scale Image 5 progress into the 10%-25% range of the visit scale
            visit_image5_prog = 0.10 + (inner_val * 0.15)
            progress(base_prog + (visit_image5_prog * v_scale), f"[{visit_name}] {inner_status}")

        try:
            ref_stack = create_reference_stack_image5(visit_dir, folder_contents, progress_callback=image5_progress_cb)
            log(f"  Image 5 reference stack created. Slices: {ref_stack.shape[0]}")
        except Exception as e:
            log(f"ERROR creating reference stack for {visit_name}: {str(e)}")
            return False

        # 3. Step 2 & 3: Registration (CLAHE tuning runs per layer on warped data — ImageJ parity)
        progress(base_prog + (0.25 * v_scale), f"[{visit_name}] Computing alignment...")

        # Integration of granular alignment progress (40% -> 70%)
        def alignment_progress_cb(inner_val, inner_status):
            # Scale alignment progress into the 40%-70% range of the visit scale
            visit_aligned_prog = 0.40 + (inner_val * 0.30)
            progress(base_prog + (visit_aligned_prog * v_scale), f"[{visit_name}] {inner_status}")

        matrices = calculate_affine_transformations(
            ref_stack, 
            progress_callback=alignment_progress_cb,
            log_callback=log
        )
        
        # [Verify Matrices Alignment]: matrices[0] should be Identity
        if not np.allclose(matrices[0], np.eye(2, 3)):
            log("  [WARN] Alignment Anchor (Slice 1) is not Identity. Results may be shifted.")

        # 4. Step 4: Apply to all layers and save results (70% -> 100%)
        sorted_captures = sorted(folder_contents.keys())
        total_captures = len(sorted_captures)
        total_layers = len(folder_contents[sorted_captures[0]])
        
        for layer_idx in range(total_layers):
            layer_weight = 1.0 / total_layers
            layer_base = 0.70 + (layer_idx * layer_weight * 0.30)
            
            log(f"  Processing Layer {layer_idx+1}/{total_layers}...")
            
            # Build original stack for this specific layer across all captures
            raw_stack = []
            for cap_idx, capture_folder in enumerate(sorted_captures):
                image_files = folder_contents[capture_folder]
                filename = image_files[layer_idx]
                
                file_path = os.path.join(visit_dir, capture_folder, filename)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
                # Match 4x enlargement
                raw_stack.append(enlarge_image_4x(img))
            
            stack_array = np.stack(raw_stack, axis=0)
            num_slices = stack_array.shape[0]

            # Nested callback for Warping (0% - 50% of layer progress)
            def layer_warp_cb(inner_val, inner_status):
                p = layer_base + (inner_val * 0.5 * layer_weight * 0.30)
                progress(base_prog + (p * v_scale), f"[{visit_name}] L{layer_idx+1}: {inner_status}")

            # Apply transformation matrices from Image 5
            registered_stack = apply_transformations_to_stack(stack_array, matrices, progress_callback=layer_warp_cb)

            # Final enhancement with optimized CLAHE (50% - 90% of layer progress)
            # ImageJ optimizes on the middle slice of each registered stack, not the Image 5 reference.
            if apply_clahe_to_ref or layer_idx > 0:
                mid_slice = (num_slices + 1) // 2 - 1
                if automate_tuning:
                    progress(
                        base_prog + ((layer_base + 0.5 * layer_weight * 0.30) * v_scale),
                        f"[{visit_name}] L{layer_idx+1}: Finding optimal CLAHE...",
                    )
                    b, h, s = optimize_clahe_parameters(registered_stack[mid_slice])
                    log(f"  Layer {layer_idx + 1} CLAHE: BlockSize={b}px, Bins={h}, Slope={s}")
                else:
                    b, h, s = (32, 256, 4.0)
                    log(f"  Layer {layer_idx + 1} CLAHE (fixed): BlockSize={b}px, Bins={h}, Slope={s}")

                for s_idx in range(num_slices):
                    clahe_p = layer_base + (0.5 * layer_weight * 0.30) + (
                        (s_idx / num_slices) * 0.4 * layer_weight * 0.30
                    )
                    progress(
                        base_prog + (clahe_p * v_scale),
                        f"[{visit_name}] L{layer_idx+1}: Enhancing {s_idx+1}/{num_slices}...",
                    )
                    registered_stack[s_idx] = apply_clahe(
                        registered_stack[s_idx], clip_limit=s, block_size=b, nbins=h
                    )
            
            # Average Intensity Projection (90% - 100%)
            progress(base_prog + ( (layer_base + 0.9 * layer_weight * 0.30) * v_scale ), f"[{visit_name}] L{layer_idx+1}: Averaging...")
            avg_img = average_project_stack(registered_stack)
            
            # Save
            output_filename = f"{patient_name}-Avg-Stack_{visit_name}_image{layer_idx+1}.tif"
            output_filepath = os.path.join(patient_output_dir, output_filename)
            tifffile.imwrite(output_filepath, avg_img)
            log(f"    Saved: {output_filename}")

    progress(1.0, "Completed successfully.")
    log("\n--- Pipeline Finished ---")
    return True

