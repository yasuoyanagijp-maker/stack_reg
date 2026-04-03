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

def run_registration_pipeline(
    input_dir: str, 
    output_dir: str, 
    apply_clahe_to_ref: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None
):
    """
    Orchestrates the entire OCTA Registration process with strict parity 
    to the ImageJ macro logic.
    
    This function:
    1. Validates the folder structure and file order.
    2. Builds the 'Image 5' reference stack (Stack of Averages).
    3. Calculates registration transformations on the reference stack.
    4. Applies transformations and optimized enhancement to all other layers.
    5. Saves averaged results as TIFFs.
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
    progress(0.05, "Validating folder structure...")
    
    # 1. Validation
    is_valid, msg, folder_contents = validate_folder_structure(input_dir)
    if not is_valid:
        log(f"ERROR: {msg}")
        return False
    log(msg)

    # Ensure output directory exists (patient folder)
    patient_name = os.path.basename(input_dir.rstrip(os.sep))
    patient_output_dir = os.path.join(output_dir, patient_name)
    if not os.path.exists(patient_output_dir):
        os.makedirs(patient_output_dir)
        log(f"Created output subdirectory: {patient_output_dir}")

    # 2. Step 1: Create 'Image 5' Reference Stack (Stack of Averages)
    progress(0.15, "Building Image 5 Reference Stack...")
    try:
        ref_stack = create_reference_stack_image5(input_dir, folder_contents)
        log(f"Image 5 reference stack created. Slices: {ref_stack.shape[0]}")
    except Exception as e:
        log(f"ERROR creating reference stack: {str(e)}")
        return False

    # 3. Step 2 & 3: Optimization & Registration Calculation
    progress(0.35, "Optimizing CLAHE parameters...")
    middle_idx = ref_stack.shape[0] // 2
    best_params = optimize_clahe_parameters(ref_stack[middle_idx])
    b, h, s = best_params
    log(f"Optimized CLAHE set to: Block={b}, Slope={s}")

    progress(0.45, "Calculating Affine registration matrices...")
    matrices = calculate_affine_transformations(ref_stack)
    log("Transformation calculation complete.")

    # 4. Step 4: Apply to all folders and save results
    sorted_folders = sorted(folder_contents.keys())
    total_folders = len(sorted_folders)
    
    for idx, folder_name in enumerate(sorted_folders):
        log(f"Processing Layer {idx+1}/{total_folders}: {folder_name}...")
        progress(0.5 + (0.4 * (idx / total_folders)), f"Processing {folder_name}...")
        
        # Build original stack for this layer
        image_files = folder_contents[folder_name]
        raw_stack = []
        for filename in image_files:
            file_path = os.path.join(input_dir, folder_name, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # Match 4x enlargement
            raw_stack.append(enlarge_image_4x(img))
        
        stack_array = np.stack(raw_stack, axis=0)
        
        # Apply transformation matrices from Image 5
        registered_stack = apply_transformations_to_stack(stack_array, matrices)
        
        # Final enhancement with optimized CLAHE
        # Matches ImageJ logic: applies to slices 2-n by default or ALL if requested
        if apply_clahe_to_ref or idx > 0:
            log(f"  Enhancing {folder_name} with optimized CLAHE...")
            for s_idx in range(registered_stack.shape[0]):
                registered_stack[s_idx] = apply_clahe(registered_stack[s_idx], clip_limit=s, tile_size=b)
        
        # Average Intensity Projection
        avg_img = average_project_stack(registered_stack)
        
        # Save as TIFF
        output_filename = f"{patient_name}-Avg-{folder_name}.tif"
        output_filepath = os.path.join(patient_output_dir, output_filename)
        tifffile.imwrite(output_filepath, avg_img)
        log(f"  Saved Average Projection: {output_filename}")

    progress(1.0, "Completed successfully.")
    log("--- Pipeline Finished ---")
    return True

