import os
from typing import List, Dict, Tuple

def validate_folder_structure(main_dir: str) -> Tuple[bool, str, Dict[str, List[str]]]:
    """
    Checks if the folders and files are organized correctly for processing.
    
    This function mirrors the logic in the original ImageJ macro but adds 
    an explicit check to ensure all image folders have the same files in the same order.
    
    Args:
        main_dir: The path to the visit directory (e.g., "Visit_1")
        
    Returns:
        A tuple of (is_valid, error_message, folder_contents_map)
    """
    
    # 1. Get the list of subfolders (e.g., Layer 1, Layer 2, etc.)
    # In ImageJ: subfolders = getFileList(subDirPath);
    try:
        subfolders = sorted([f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))])
    except Exception as e:
        return False, f"Could not read directory: {str(e)}", {}

    if not subfolders:
        return False, "No subfolders found in the selected directory.", {}

    folder_contents = {}
    reference_files = []
    reference_folder = ""

    # 2. Iterate through each subfolder to identify files
    for sf in subfolders:
        current_folder_path = os.path.join(main_dir, sf)
        
        # Get list of image files (tif or jpg as per original macro)
        # In ImageJ: list = getFileList(currentFolder);
        files = sorted([f for f in os.listdir(current_folder_path) 
                       if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg'))])
        
        if len(files) < 2:
            return False, f"Folder '{sf}' must contain at least 2 images for registration.", {}
        
        # 3. Consistency Check (New requirement)
        # Ensure that every folder has the exact same number of files and matching names/order
        if not reference_files:
            # This is the first folder we check; use it as the reference for others
            reference_files = files
            reference_folder = sf
        else:
            if len(files) != len(reference_files):
                return False, (f"Inconsistency found: Folder '{sf}' has {len(files)} files, "
                               f"but folder '{reference_folder}' has {len(reference_files)} files. "
                               "All folders must contain the same number of images."), {}
            
            # Check if all filenames match exactly in order
            for i, (f_ref, f_curr) in enumerate(zip(reference_files, files)):
                if f_ref != f_curr:
                    return False, (f"Inconsistency found: File index {i+1} follows a different naming pattern. "
                                   f"Expected '{f_ref}' from folder '{reference_folder}', "
                                   f"but found '{f_curr}' in folder '{sf}'. "
                                   "All folders must have matching files in the same order."), {}
        
        folder_contents[sf] = files

    # If we reached here, everything is consistent
    summary_message = (f"Validation successful: {len(subfolders)} folders found, "
                       f"each containing {len(reference_files)} matching images.")
    
    return True, summary_message, folder_contents

if __name__ == "__main__":
    # Test logic
    import sys
    if len(sys.argv) > 1:
        valid, msg, contents = validate_folder_structure(sys.argv[1])
        print(f"Valid: {valid}")
        print(f"Message: {msg}")
    else:
        print("Please provide a directory path to test.")
