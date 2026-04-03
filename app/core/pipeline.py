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
        progress(base_prog + (0.15 * v_scale), f"[{visit_name}] Building Reference Stack...")
        try:
            ref_stack = create_reference_stack_image5(visit_dir, folder_contents)
            log(f"  Image 5 reference stack created. Slices: {ref_stack.shape[0]}")
        except Exception as e:
            log(f"ERROR creating reference stack for {visit_name}: {str(e)}")
            return False

        # 3. Step 2 & 3: Optimization & Registration Calculation
        progress(base_prog + (0.35 * v_scale), f"[{visit_name}] Optimizing CLAHE...")
        middle_idx = ref_stack.shape[0] // 2
        best_params = optimize_clahe_parameters(ref_stack[middle_idx])
        b, h, s = best_params
        log(f"  Optimized CLAHE set to: Block={b}, Slope={s}")

        progress(base_prog + (0.45 * v_scale), f"[{visit_name}] Calculating Affine matrices...")
        matrices = calculate_affine_transformations(ref_stack)
        log("  Transformation calculation complete.")

        # 4. Step 4: Apply to all layers and save results
        sorted_captures = sorted(folder_contents.keys())
        total_captures = len(sorted_captures)
        total_layers = len(folder_contents[sorted_captures[0]])
        
        for layer_idx in range(total_layers):
            log(f"  Processing Layer {layer_idx+1}/{total_layers}...")
            progress(base_prog + ((0.5 + (0.4 * (layer_idx / total_layers))) * v_scale), f"[{visit_name}] Processing Layer {layer_idx+1}...")
            
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
            
            # Apply transformation matrices from Image 5
            registered_stack = apply_transformations_to_stack(stack_array, matrices)
            
            # Final enhancement with optimized CLAHE
            # Matches ImageJ logic: applies to slices 2-n by default or ALL if requested
            if apply_clahe_to_ref or layer_idx > 0:
                for s_idx in range(registered_stack.shape[0]):
                    registered_stack[s_idx] = apply_clahe(registered_stack[s_idx], clip_limit=s, tile_size=b)
            
            # Average Intensity Projection
            avg_img = average_project_stack(registered_stack)
            
            # Formulate output filename matching the ImageJ manual format exactly
            # Format: 患者名-Avg-Stack_Visit1_image1.tif
            output_filename = f"{patient_name}-Avg-Stack_{visit_name}_image{layer_idx+1}.tif"
            output_filepath = os.path.join(patient_output_dir, output_filename)
            tifffile.imwrite(output_filepath, avg_img)
            log(f"    Saved: {output_filename}")

    progress(1.0, "Completed successfully.")
    log("\n--- Pipeline Finished ---")
    return True

