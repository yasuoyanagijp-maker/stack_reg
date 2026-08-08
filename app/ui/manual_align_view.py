import flet as ft
import numpy as np

from app.core.registration import (
    estimate_affine_from_correspondences,
    compute_alignment_cc,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from app.core.manual_align import to_png_bytes, make_overlay, draw_landmarks

DISPLAY_W = 360

def _placeholder_png(text: str = "No capture selected") -> bytes:
    """
    A neutral DISPLAY_W x DISPLAY_W panel shown in the image slots before a
    capture is selected. Keeping the images always visible (rather than
    ``visible=False``) preserves the editor layout: collapsing them makes the
    whole panel (captions, buttons, status) fail to render in Flet 0.84.
    An empty ``src`` is also not an option - it renders an error box.
    """
    import cv2
    canvas = np.full((DISPLAY_W, DISPLAY_W), 24, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.putText(canvas, text, ((DISPLAY_W - tw) // 2, (DISPLAY_W + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 110, 1, cv2.LINE_AA)
    return to_png_bytes(canvas)


_PLACEHOLDER_PNG = _placeholder_png()
_PREVIEW_PLACEHOLDER_PNG = _placeholder_png("No preview computed")


def _short_name(name: str, limit: int = 24) -> str:
    """Truncate long capture folder names for compact UI labels."""
    return name if len(name) <= limit else name[:limit - 1] + "…"


def _capture_folder(plan, idx: int) -> str:
    """Folder name of capture ``idx`` in ``plan`` ('' if unknown)."""
    caps = getattr(plan, "sorted_captures", None) or []
    return caps[idx] if 0 <= idx < len(caps) else ""


def _resize_for_display(img: np.ndarray, width: int = DISPLAY_W):
    """Return (display_grayscale, scale) where scale maps display -> full coords."""
    import cv2
    h, w = img.shape[:2]
    disp_h = max(1, int(round(h * width / w)))
    disp = cv2.resize(img, (width, disp_h), interpolation=cv2.INTER_AREA)
    scale = w / float(width)
    return disp, disp_h, scale


def create_manual_align_view(
    page: ft.Page,
    plans,
    on_back,
    on_finalize,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
):
    """
    Interactive corresponding-point correction screen.

    ``plans``       : list of ``VisitPlan`` (already prepared / auto-aligned).
    ``on_back``     : callback to return to the dashboard.
    ``on_finalize`` : callback(overrides_by_visit, points_by_visit) invoked when
                      the user commits the corrections; the dashboard runs the
                      warping/averaging in a worker thread.
    """
    plans_by_name = {p.visit_name: p for p in plans}

    # --- Correction state -------------------------------------------------
    # overrides[visit][capture_idx] = 2x3 float32 matrix
    overrides = {p.visit_name: {} for p in plans}
    # points[(visit, capture_idx)] = {"ref": [[x,y]...], "src": [[x,y]...]} (full-image coords)
    points = {}

    state = {"visit": plans[0].visit_name, "capture": None, "disp_scale": 1.0}

    # --- Controls ---------------------------------------------------------
    visit_row = ft.Row(wrap=True, spacing=6)
    capture_list = ft.ListView(expand=True, spacing=4, padding=4)

    ref_img = ft.Image(width=DISPLAY_W, height=DISPLAY_W, fit=ft.BoxFit.FILL,
                       border_radius=6, src=_PLACEHOLDER_PNG)
    src_img = ft.Image(width=DISPLAY_W, height=DISPLAY_W, fit=ft.BoxFit.FILL,
                       border_radius=6, src=_PLACEHOLDER_PNG)
    preview_img = ft.Image(width=DISPLAY_W, height=DISPLAY_W, fit=ft.BoxFit.FILL,
                           border_radius=6, src=_PREVIEW_PLACEHOLDER_PNG)

    status_text = ft.Text("Select a capture on the left to begin.",
                          size=13, color=ft.Colors.CYAN_200)
    cc_text = ft.Text("", size=13, color=ft.Colors.GREY_300)

    ref_caption = ft.Text(
        f"Reference (Capture 1 — {_short_name(_capture_folder(plans[0], 0))})",
        size=13, weight=ft.FontWeight.BOLD)
    src_caption = ft.Text("Source", size=13, weight=ft.FontWeight.BOLD)

    def cur_points():
        key = (state["visit"], state["capture"])
        if key not in points:
            points[key] = {"ref": [], "src": []}
        return points[key]

    def to_display_pts(full_pts):
        s = state["disp_scale"]
        return [[x / s, y / s] for (x, y) in full_pts]

    def render_images():
        plan = plans_by_name[state["visit"]]
        cap = state["capture"]
        if cap is None:
            return
        ref_full = plan.ref_stack[0]
        src_full = plan.ref_stack[cap]
        ref_disp, disp_h, scale = _resize_for_display(ref_full)
        src_disp, _, _ = _resize_for_display(src_full)
        state["disp_scale"] = scale
        ref_img.height = disp_h
        src_img.height = disp_h
        preview_img.height = disp_h

        pts = cur_points()
        ref_marked = draw_landmarks(ref_disp, to_display_pts(pts["ref"]), color=(0, 255, 0))
        src_marked = draw_landmarks(src_disp, to_display_pts(pts["src"]), color=(255, 0, 255))
        ref_img.src = to_png_bytes(ref_marked)
        src_img.src = to_png_bytes(src_marked)
        ref_caption.value = (
            f"Reference (Capture 1 — {_short_name(_capture_folder(plan, 0))}) — green"
        )
        src_caption.value = (
            f"Source (Capture {cap+1} — {_short_name(_capture_folder(plan, cap))}) — magenta"
        )
        page.update()

    def refresh_capture_list():
        plan = plans_by_name[state["visit"]]
        capture_list.controls.clear()
        for idx in range(1, len(plan.matrices)):  # skip anchor (capture 0)
            score = plan.scores[idx] if idx < len(plan.scores) else 0.0
            low = score < threshold
            corrected = idx in overrides[plan.visit_name]
            if corrected:
                badge, col = "corrected", ft.Colors.GREEN_400
            elif low:
                badge, col = f"{score:.2f} low", ft.Colors.RED_400
            else:
                badge, col = f"{score:.2f}", ft.Colors.GREY_400
            selected = idx == state["capture"]
            capture_list.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Icon(
                            ft.Icons.CHECK_CIRCLE if corrected else
                            (ft.Icons.WARNING_AMBER if low else ft.Icons.CIRCLE_OUTLINED),
                            color=col, size=16,
                        ),
                        ft.Text(f"Capture {idx+1} — {_short_name(_capture_folder(plan, idx))}",
                                size=13,
                                weight=ft.FontWeight.BOLD if selected else ft.FontWeight.NORMAL),
                        ft.Container(expand=True),
                        ft.Text(badge, size=11, color=col),
                    ], spacing=6),
                    padding=8,
                    border_radius=6,
                    bgcolor=ft.Colors.CYAN_900 if selected else ft.Colors.GREY_900,
                    on_click=lambda _, i=idx: select_capture(i),
                )
            )
        page.update()

    def select_capture(idx):
        state["capture"] = idx
        preview_img.src = _PREVIEW_PLACEHOLDER_PNG
        cc_text.value = ""
        n_ref = len(cur_points()["ref"])
        n_src = len(cur_points()["src"])
        status_text.value = (
            f"Capture {idx+1}: click matching landmarks — same feature on both images "
            f"(≥3 pairs). Points: ref {n_ref}, src {n_src}."
        )
        refresh_capture_list()
        render_images()

    def select_visit(name):
        state["visit"] = name
        state["capture"] = None
        refresh_capture_list()
        ref_img.src = _PLACEHOLDER_PNG
        src_img.src = _PLACEHOLDER_PNG
        preview_img.src = _PREVIEW_PLACEHOLDER_PNG
        ref_img.height = DISPLAY_W
        src_img.height = DISPLAY_W
        preview_img.height = DISPLAY_W
        plan = plans_by_name[name]
        ref_caption.value = f"Reference (Capture 1 — {_short_name(_capture_folder(plan, 0))})"
        src_caption.value = "Source"
        status_text.value = "Select a capture on the left to begin."
        page.update()

    def add_point(which, e):
        if state["capture"] is None:
            return
        pos = getattr(e, "local_position", None)
        if pos is None:
            return
        s = state["disp_scale"]
        fx, fy = pos.x * s, pos.y * s
        cur_points()[which].append([fx, fy])
        n_ref = len(cur_points()["ref"])
        n_src = len(cur_points()["src"])
        status_text.value = (
            f"Points: ref {n_ref}, src {n_src}. "
            + ("Ready to compute." if (n_ref == n_src and n_ref >= 3)
               else "Add matching pairs (≥3, equal counts).")
        )
        render_images()

    def undo_point(which, e):
        if state["capture"] is None:
            return
        pts = cur_points()[which]
        if pts:
            pts.pop()
        render_images()

    def reset_points(e):
        if state["capture"] is None:
            return
        key = (state["visit"], state["capture"])
        points[key] = {"ref": [], "src": []}
        preview_img.src = _PREVIEW_PLACEHOLDER_PNG
        cc_text.value = ""
        status_text.value = "Points cleared."
        render_images()

    def compute_preview(e):
        if state["capture"] is None:
            return
        plan = plans_by_name[state["visit"]]
        cap = state["capture"]
        pts = cur_points()
        n = min(len(pts["ref"]), len(pts["src"]))
        if n < 3:
            status_text.value = "Need at least 3 matching point pairs."
            page.update()
            return
        try:
            matrix = estimate_affine_from_correspondences(
                np.array(pts["ref"][:n]), np.array(pts["src"][:n])
            )
        except ValueError as ex:
            status_text.value = f"Cannot compute: {ex}"
            page.update()
            return

        ref_full = plan.ref_stack[0]
        src_full = plan.ref_stack[cap]
        overlay = make_overlay(ref_full, src_full, matrix)
        auto_cc = compute_alignment_cc(ref_full, src_full, plan.matrices[cap])
        manual_cc = compute_alignment_cc(ref_full, src_full, matrix)

        preview_img.src = to_png_bytes(overlay, max_side=DISPLAY_W)
        cc_text.value = (
            f"Alignment score — auto: {auto_cc:.3f}   manual: {manual_cc:.3f}   "
            + ("(better ✓)" if manual_cc >= auto_cc else "(worse — check points)")
        )
        cc_text.color = ft.Colors.GREEN_400 if manual_cc >= auto_cc else ft.Colors.AMBER_400
        state["_pending_matrix"] = matrix
        status_text.value = "Preview ready. Green = reference, magenta = source; white = aligned."
        page.update()

    def accept_correction(e):
        mat = state.get("_pending_matrix")
        if mat is None or state["capture"] is None:
            status_text.value = "Compute a preview before accepting."
            page.update()
            return
        overrides[state["visit"]][state["capture"]] = mat
        status_text.value = f"Capture {state['capture']+1} correction accepted."
        refresh_capture_list()
        page.update()

    def revert_auto(e):
        if state["capture"] is None:
            return
        overrides[state["visit"]].pop(state["capture"], None)
        state.pop("_pending_matrix", None)
        preview_img.src = _PREVIEW_PLACEHOLDER_PNG
        cc_text.value = ""
        status_text.value = f"Capture {state['capture']+1} reverted to automatic alignment."
        refresh_capture_list()
        page.update()

    def finalize_click(e):
        clean_overrides = {v: dict(d) for v, d in overrides.items() if d}
        clean_points = {}
        for (visit, cap), pt in points.items():
            if visit in clean_overrides and cap in clean_overrides[visit]:
                clean_points.setdefault(visit, {})[cap] = pt
        on_finalize(clean_overrides, clean_points)

    # --- Build visit selector --------------------------------------------
    for p in plans:
        btn = ft.FilledTonalButton(
            content=p.visit_name,
            on_click=lambda _, n=p.visit_name: select_visit(n),
        )
        btn.data = p.visit_name
        visit_row.controls.append(btn)

    ref_gd = ft.GestureDetector(
        content=ref_img,
        on_tap_down=lambda e: add_point("ref", e),
    )
    src_gd = ft.GestureDetector(
        content=src_img,
        on_tap_down=lambda e: add_point("src", e),
    )

    control_row = ft.Row([
        ft.OutlinedButton("Undo ref", icon=ft.Icons.UNDO,
                          on_click=lambda e: undo_point("ref", e)),
        ft.OutlinedButton("Undo src", icon=ft.Icons.UNDO,
                          on_click=lambda e: undo_point("src", e)),
        ft.OutlinedButton("Reset", icon=ft.Icons.CLEAR, on_click=reset_points),
        ft.FilledButton("Compute & Preview", icon=ft.Icons.CALCULATE, on_click=compute_preview),
        ft.FilledButton("Accept", icon=ft.Icons.CHECK, on_click=accept_correction),
        ft.OutlinedButton("Revert to auto", icon=ft.Icons.RESTORE, on_click=revert_auto),
    ], wrap=True, spacing=8)

    left_panel = ft.Container(
        content=ft.Column([
            ft.Text("Visits", size=14, weight=ft.FontWeight.BOLD),
            visit_row,
            ft.Divider(height=10),
            ft.Text("Captures (aligned to Capture 1)", size=14, weight=ft.FontWeight.BOLD),
            ft.Text(f"Red = below confidence {threshold:.2f}", size=11, color=ft.Colors.RED_300),
            capture_list,
        ], spacing=8, expand=True),
        width=260,
        padding=8,
    )

    editor_panel = ft.Column([
        ft.Row([
            ft.Column([ref_caption, ft.Container(content=ref_gd,
                       border=ft.Border.all(1, ft.Colors.GREEN_700), border_radius=6)]),
            ft.Column([src_caption, ft.Container(content=src_gd,
                       border=ft.Border.all(1, ft.Colors.PURPLE_400), border_radius=6)]),
            ft.Column([ft.Text("Overlay preview", size=13, weight=ft.FontWeight.BOLD),
                       ft.Container(content=preview_img,
                       border=ft.Border.all(1, ft.Colors.CYAN_700), border_radius=6)]),
        ], spacing=16, scroll=ft.ScrollMode.AUTO),
        control_row,
        status_text,
        cc_text,
    ], spacing=12, expand=True, scroll=ft.ScrollMode.AUTO)

    header = ft.Row([
        ft.IconButton(icon=ft.Icons.ARROW_BACK, tooltip="Back to dashboard",
                      on_click=lambda e: on_back()),
        ft.Text("Manual Corresponding-Point Correction", size=22,
                weight=ft.FontWeight.BOLD, color=ft.Colors.CYAN_400),
        ft.Container(expand=True),
        ft.FilledButton("Finalize & Save", icon=ft.Icons.SAVE,
                        style=ft.ButtonStyle(bgcolor=ft.Colors.CYAN_700, color=ft.Colors.WHITE),
                        on_click=finalize_click),
    ])

    view = ft.Column([
        header,
        ft.Divider(height=10),
        ft.Row([left_panel, ft.VerticalDivider(width=1), editor_panel], expand=True),
    ], expand=True, spacing=8)

    # Initialize selection
    refresh_capture_list()
    return view
