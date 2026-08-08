"""
Post-finalize results gallery: shows the averaged .tif images produced in the
patient output directory so the user can review them without leaving the app.

Flet/Flutter cannot render TIFF files via ``Image(src=path)``, so each file is
loaded with tifffile and converted to PNG bytes (downscaled for display).
"""
import os

import numpy as np
import flet as ft

from app.core.manual_align import to_png_bytes

DISPLAY_MAX_SIDE = 700


def _load_display_png(path: str, max_side: int = DISPLAY_MAX_SIDE) -> bytes:
    import tifffile
    img = np.asarray(tifffile.imread(path))
    if img.ndim == 3:
        img = img[0]
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img.astype(np.float32) - lo) / max(hi - lo, 1e-9) * 255.0).astype(np.uint8)
    return to_png_bytes(img, max_side=max_side)


def create_results_view(page: ft.Page, patient_output_dir: str, on_back,
                        corrections_summary=None):
    """
    Scrollable gallery of the generated averaged .tif images.

    ``corrections_summary`` optionally maps visit name -> list of capture
    indices (0-based) that were manually corrected, shown above the gallery.
    """
    try:
        tif_files = sorted(
            f for f in os.listdir(patient_output_dir)
            if f.lower().endswith((".tif", ".tiff"))
        )
    except OSError:
        tif_files = []

    gallery = ft.ListView(expand=True, spacing=18, padding=12)
    for fname in tif_files:
        path = os.path.join(patient_output_dir, fname)
        try:
            body = ft.Image(src=_load_display_png(path), fit=ft.BoxFit.CONTAIN,
                            border_radius=6)
        except Exception as ex:
            body = ft.Text(f"Could not load {fname}: {ex}",
                           color=ft.Colors.RED_400, size=12)
        gallery.controls.append(
            ft.Column([
                ft.Text(fname, size=13, weight=ft.FontWeight.BOLD,
                        color=ft.Colors.CYAN_200),
                body,
            ], spacing=6)
        )
    if not tif_files:
        gallery.controls.append(
            ft.Text("No generated .tif images found in the output folder.",
                    color=ft.Colors.AMBER_400)
        )

    summary_lines = []
    if corrections_summary:
        for visit, caps in sorted(corrections_summary.items()):
            caps_h = ", ".join(str(c + 1) for c in sorted(caps))
            summary_lines.append(
                ft.Text(f"{visit}: Capture {caps_h} manually corrected",
                        size=12, color=ft.Colors.GREEN_300)
            )
    else:
        summary_lines.append(
            ft.Text("No manual corrections were applied (automatic alignment only).",
                    size=12, color=ft.Colors.GREY_400)
        )

    header = ft.Row([
        ft.IconButton(icon=ft.Icons.ARROW_BACK, tooltip="Back to dashboard",
                      on_click=lambda e: on_back()),
        ft.Text("Generated Results", size=22, weight=ft.FontWeight.BOLD,
                color=ft.Colors.CYAN_400),
        ft.Container(expand=True),
        ft.Text(patient_output_dir, size=11, color=ft.Colors.GREY_500),
    ])

    return ft.Column([
        header,
        ft.Divider(height=10),
        *summary_lines,
        gallery,
    ], expand=True, spacing=8)
