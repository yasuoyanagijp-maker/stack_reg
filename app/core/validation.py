import os
from typing import List, Dict, Tuple

from app.core.image_proc import imread_grayscale

def validate_folder_structure(main_dir: str) -> Tuple[bool, str, Dict[str, List[str]]]:
    """
    Checks if the Capture folders (e.g., 1/, 2/, 3/) contain the correct layers.
    Returns:
        (is_valid, msg, list_of_capture_folders)
    """
    try:
        subfolders = sorted([f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))])
    except Exception as e:
        return False, f"Could not read directory: {str(e)}", {}

    if not subfolders:
        return False, "No subfolders found in the selected directory.", {}

    folder_contents = {}
    reference_count = 0
    reference_folder = ""

    for sf in subfolders:
        current_folder_path = os.path.join(main_dir, sf)
        files = sorted([f for f in os.listdir(current_folder_path) 
                       if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg'))])
        
        if len(files) < 2:
            return False, f"Folder '{sf}' must contain at least 2 images for registration.", {}
        
        if reference_count == 0:
            reference_count = len(files)
            reference_folder = sf
        else:
            if len(files) != reference_count:
                return False, (f"Inconsistency found: Folder '{sf}' has {len(files)} files, "
                               f"but folder '{reference_folder}' has {reference_count} files. "
                               "All folders must contain the same number of images."), {}
        
        folder_contents[sf] = files

    summary_message = (f"Validation successful: {len(subfolders)} captures found, "
                       f"each containing {reference_count} layers.")
    
    return True, summary_message, folder_contents


def validate_images_readable(
    main_dir: str, folder_contents: Dict[str, List[str]], max_report: int = 5
) -> Tuple[bool, str]:
    """
    Verify every listed image can be decoded (catches Windows Unicode path issues early).
    """
    unreadable: List[str] = []
    for folder_name, files in folder_contents.items():
        for filename in files:
            path = os.path.join(main_dir, folder_name, filename)
            if imread_grayscale(path) is None:
                unreadable.append(path)

    if not unreadable:
        return True, "All images are readable."

    sample = unreadable[:max_report]
    extra = len(unreadable) - len(sample)
    lines = "\n  ".join(sample)
    suffix = f"\n  ... and {extra} more." if extra > 0 else ""
    return False, (
        f"Cannot read {len(unreadable)} image(s). "
        "Check that files exist and paths are valid:\n  "
        f"{lines}{suffix}"
    )


if __name__ == "__main__":
    # Test logic
    import sys
    if len(sys.argv) > 1:
        valid, msg, contents = validate_folder_structure(sys.argv[1])
        print(f"Valid: {valid}")
        print(f"Message: {msg}")
    else:
        print("Please provide a directory path to test.")
