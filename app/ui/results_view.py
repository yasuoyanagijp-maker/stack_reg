"""
Post-registration review screen: after the main stacking pipeline finishes, the
user visually inspects averaged stack images (image1–image4) via a dropdown
and may start manual corresponding-point correction for the selected layer.
"""
import os
import re
from collections import defaultdict

import numpy as np
import flet as ft

from app.core.manual_align import to_png_bytes

DISPLAY_MAX_SIDE = 700

_IMAGE_RE = re.compile(r"_image(\d+)\.(tif|tiff)$", re.IGNORECASE)
_VISIT_RE = re.compile(r"Avg-Stack_(.+)_image\d+\.(tif|tiff)$", re.IGNORECASE)

_EMPTY_PNG = to_png_bytes(np.full((8, 8), 24, dtype=np.uint8))


def _load_display_png(path: str, max_side: int = DISPLAY_MAX_SIDE) -> bytes:
    import tifffile
    img = np.asarray(tifffile.imread(path))
    if img.ndim == 3:
        img = img[0]
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img.astype(np.float32) - lo) / max(hi - lo, 1e-9) * 255.0).astype(np.uint8)
    return to_png_bytes(img, max_side=max_side)


def _index_output_tifs(patient_output_dir: str):
    """
    Index generated averaged stacks as
    ``{image_num: {visit_name: filepath}}``.

    Filenames follow ``{patient}-Avg-Stack_{visit}_image{N}.tif``.
    """
    by_image = defaultdict(dict)
    try:
        names = os.listdir(patient_output_dir)
    except OSError:
        return by_image

    for fname in sorted(names):
        m = _IMAGE_RE.search(fname)
        if not m:
            continue
        image_num = int(m.group(1))
        vm = _VISIT_RE.search(fname)
        visit = vm.group(1) if vm else "Visit"
        by_image[image_num][visit] = os.path.join(patient_output_dir, fname)
    return by_image


def create_results_view(
    page: ft.Page,
    patient_output_dir: str,
    on_back,
    on_review_correct=None,
    corrections_summary=None,
    initial_image_num: int | None = None,
):
    """
    Visual review of generated averaged .tif images.

    Users pick ``image1`` … ``imageN`` from a dropdown (and a visit when several
    exist). ``on_review_correct(image_num, visit_name)`` starts manual
    registration for the currently selected layer — there is no automatic
    bad-overlay detection.
    """
    by_image = _index_output_tifs(patient_output_dir)
    image_nums = sorted(by_image.keys())

    preview = ft.Image(src=_EMPTY_PNG, fit=ft.BoxFit.CONTAIN, border_radius=6)
    filename_label = ft.Text("", size=13, weight=ft.FontWeight.BOLD, color=ft.Colors.CYAN_200)
    status = ft.Text("", size=12, color=ft.Colors.GREY_400)

    state = {
        "image_num": initial_image_num if initial_image_num in by_image else (image_nums[0] if image_nums else None),
        "visit": None,
    }

    visit_dd = ft.Dropdown(
        label="Visit",
        width=280,
        options=[],
        visible=False,
    )
    image_dd = ft.Dropdown(
        label="Result image",
        width=200,
        options=[
            ft.DropdownOption(key=str(n), text=f"image{n}")
            for n in image_nums
        ],
        value=str(state["image_num"]) if state["image_num"] is not None else None,
    )

    def _visits_for_image(image_num):
        return sorted(by_image.get(image_num, {}).keys())

    def refresh_preview():
        image_num = state["image_num"]
        if image_num is None:
            preview.src = _EMPTY_PNG
            filename_label.value = ""
            status.value = "No generated .tif images found in the output folder."
            status.color = ft.Colors.AMBER_400
            page.update()
            return

        visits = _visits_for_image(image_num)
        if not visits:
            status.value = f"No files for image{image_num}."
            status.color = ft.Colors.AMBER_400
            page.update()
            return

        if state["visit"] not in visits:
            state["visit"] = visits[0]

        visit_dd.options = [ft.DropdownOption(key=v, text=v) for v in visits]
        visit_dd.value = state["visit"]
        visit_dd.visible = len(visits) > 1

        path = by_image[image_num][state["visit"]]
        try:
            preview.src = _load_display_png(path)
            filename_label.value = os.path.basename(path)
            status.value = (
                "Visually inspect the stack. Choose image1–image4, then Review & Correct "
                "to open manual alignment for that image. Corrected parameters are applied "
                "to all result images on Finalize."
            )
            status.color = ft.Colors.GREY_400
        except Exception as ex:
            preview.src = _EMPTY_PNG
            filename_label.value = os.path.basename(path)
            status.value = f"Could not load image: {ex}"
            status.color = ft.Colors.RED_400
        page.update()

    def on_image_change(e):
        if image_dd.value is None:
            return
        state["image_num"] = int(image_dd.value)
        state["visit"] = None
        refresh_preview()

    def on_visit_change(e):
        if visit_dd.value is None:
            return
        state["visit"] = visit_dd.value
        refresh_preview()

    image_dd.on_select = on_image_change
    visit_dd.on_select = on_visit_change

    def review_click(e):
        if on_review_correct is None or state["image_num"] is None:
            return
        on_review_correct(state["image_num"], state["visit"])

    summary_lines = []
    if corrections_summary:
        for visit, caps in sorted(corrections_summary.items()):
            caps_h = ", ".join(str(c + 1) for c in sorted(caps))
            summary_lines.append(
                ft.Text(f"{visit}: Capture {caps_h} manually corrected",
                        size=12, color=ft.Colors.GREEN_300)
            )

    review_btn = ft.FilledButton(
        "Review & Correct",
        icon=ft.Icons.TUNE,
        style=ft.ButtonStyle(bgcolor=ft.Colors.CYAN_700, color=ft.Colors.WHITE),
        tooltip="Start manual corresponding-point correction for the selected result image.",
        on_click=review_click,
        visible=on_review_correct is not None and bool(image_nums),
    )

    header = ft.Row([
        ft.IconButton(icon=ft.Icons.ARROW_BACK, tooltip="Back to dashboard",
                      on_click=lambda e: on_back()),
        ft.Text("Review Stack Results", size=22, weight=ft.FontWeight.BOLD,
                color=ft.Colors.CYAN_400),
        ft.Container(expand=True),
        ft.Text(patient_output_dir, size=11, color=ft.Colors.GREY_500),
    ])

    controls_row = ft.Row([
        image_dd,
        visit_dd,
        review_btn,
    ], spacing=16, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    body = ft.Column([
        filename_label,
        ft.Container(
            content=preview,
            bgcolor=ft.Colors.GREY_900,
            border_radius=8,
            padding=8,
            border=ft.Border.all(1, ft.Colors.CYAN_900),
            expand=True,
            alignment=ft.Alignment.CENTER,
        ),
        status,
    ], spacing=8, expand=True)

    view = ft.Column([
        header,
        ft.Divider(height=10),
        *summary_lines,
        controls_row,
        body,
    ], expand=True, spacing=8)

    refresh_preview()
    return view
