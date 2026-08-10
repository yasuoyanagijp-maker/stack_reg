import flet as ft
import numpy as np

from app.core.registration import (
    estimate_affine_from_correspondences,
    compute_alignment_cc,
    seed_correspondences_from_matrix,
    extract_feature_correspondences,
    correspondence_residuals,
    filter_correspondences_by_residual,
    nudge_affine_matrix,
)
from app.core.pipeline import realign_plan_to_reference
from app.core.manual_align import (
    to_png_bytes,
    make_overlay,
    draw_landmarks,
    draw_diagnostic_matches,
    nearest_landmark_index,
)

DISPLAY_W_MAX = 360
DISPLAY_W_MIN = 160
LEFT_PANEL_W = 280
# Divider + row spacing + page chrome reserved outside the three image panels.
_EDITOR_GUTTER = 80
# Hit radius in *display* pixels for selecting / dragging a pin.
_PIN_HIT_PX = 14.0
# Legacy alias used for placeholder tile generation.
DISPLAY_W = DISPLAY_W_MAX


def fit_display_width(page_width: float | None, *, n_images: int = 3) -> int:
    """
    Choose a per-image display width so reference / source / overlay fit in
    ``page_width`` beside the left capture list (no horizontal clipping).
    """
    pw = float(page_width) if page_width and page_width > 0 else 1100.0
    avail = pw - LEFT_PANEL_W - _EDITOR_GUTTER
    w = int(avail // max(1, n_images))
    return max(DISPLAY_W_MIN, min(DISPLAY_W_MAX, w))


def _placeholder_png(text: str = "No capture selected", side: int = DISPLAY_W) -> bytes:
    """
    A neutral square panel shown in the image slots before a capture is
    selected. Keeping the images always visible (rather than ``visible=False``)
    preserves the editor layout: collapsing them makes the whole panel
    (captions, buttons, status) fail to render in Flet 0.84.
    An empty ``src`` is also not an option - it renders an error box.
    """
    import cv2
    side = max(64, int(side))
    canvas = np.full((side, side), 24, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.putText(canvas, text, ((side - tw) // 2, (side + th) // 2),
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


def _capture_folder_suffix(plan, idx: int) -> str:
    """
    `` — {folder}`` only when the folder name differs from the capture number.

    Numbered folders (``1``, ``2``, …) matching ``idx+1`` omit the suffix so
    labels stay ``Capture 1`` instead of ``Capture 1 — 1``.
    """
    folder = _capture_folder(plan, idx)
    if not folder or folder == str(idx + 1):
        return ""
    return f" — {_short_name(folder)}"


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
    focus_layer: int | None = None,
    preloaded_stacks: dict | None = None,
):
    """
    Interactive corresponding-point correction screen.

    ``plans``        : list of ``VisitPlan`` (already prepared / auto-aligned).
    ``on_back``      : callback to return to the previous screen.
    ``on_finalize``  : callback(overrides_by_visit, points_by_visit) invoked when
                       the user commits the corrections; the dashboard runs the
                       warping/averaging in a worker thread.
    ``focus_layer``  : optional 0-based layer index (image1 → 0). When set, the
                       editor shows that layer's capture images instead of the
                       Image 5 reference stack, matching the result the user
                       selected for review.
    ``preloaded_stacks`` : optional ``{visit_name: stack}`` to avoid loading
                       4×-enlarged layers on the UI thread after navigation.
    """
    plans_by_name = {p.visit_name: p for p in plans}

    # Cache per-visit display stacks (Image 5 or the focused layer).
    display_stacks = dict(preloaded_stacks or {})

    def display_stack_for(plan):
        key = plan.visit_name
        if key not in display_stacks:
            if focus_layer is not None:
                display_stacks[key] = plan.load_layer_stack(focus_layer)
            else:
                display_stacks[key] = plan.ref_stack
        return display_stacks[key]

    # --- Correction state -------------------------------------------------
    # overrides[visit][capture_idx] = 2x3 float32 matrix
    overrides = {p.visit_name: {} for p in plans}
    # excluded[visit] = set of capture indices omitted from the average
    excluded = {p.visit_name: set() for p in plans}
    # points[(visit, capture_idx)] = {"ref": [[x,y]...], "src": [[x,y]...]} (full-image coords)
    # Manual pin landmarks (proposal 2) — typically 3–8 editable pairs.
    points = {}
    # diag[(visit, capture)] = {"ref": Nx2, "src": Nx2} automatic ORB matches (proposal 1)
    diag = {}
    # Per-visit alignment reference (0-based). Default Capture 1 / index 0.
    ref_by_visit = {
        p.visit_name: int(getattr(p, "reference_idx", 0)) for p in plans
    }
    initial_ref_by_visit = dict(ref_by_visit)

    _page_w = getattr(page, "width", None)
    if not _page_w:
        _page_w = getattr(getattr(page, "window", None), "width", None)
    initial_w = fit_display_width(_page_w)
    _prev_on_resize = page.on_resize
    state = {
        "visit": plans[0].visit_name,
        "capture": None,
        "disp_scale": 1.0,
        "disp_w": initial_w,
        "residual_px": 8.0,
        "show_diag": True,
        "nudge_step": 2.0,
        # {"which": "ref"|"src", "idx": int} while dragging / after select
        "selected": None,
        "dragging": False,
        "suppress_add": False,
    }

    def _ref_idx(visit: str | None = None) -> int:
        return int(ref_by_visit[visit or state["visit"]])

    def _ref_label(plan, ref_idx: int | None = None) -> str:
        r = _ref_idx(plan.visit_name) if ref_idx is None else int(ref_idx)
        return f"Reference (Capture {r + 1}{_capture_folder_suffix(plan, r)})"

    # --- Controls ---------------------------------------------------------
    visit_row = ft.Row(wrap=True, spacing=6)
    capture_list = ft.ListView(expand=True, spacing=4, padding=4)
    ref_dropdown = ft.Dropdown(
        label="Alignment reference",
        width=LEFT_PANEL_W - 24,
        options=[
            ft.DropdownOption(
                key=str(i),
                text=f"Capture {i + 1}{_capture_folder_suffix(plans[0], i)}",
            )
            for i in range(len(plans[0].matrices))
        ],
        value=str(_ref_idx()),
        enable_search=False,
        # on_select wired after change_reference is defined
    )

    ref_img = ft.Image(width=initial_w, height=initial_w, fit=ft.BoxFit.FILL,
                       border_radius=6, src=_PLACEHOLDER_PNG)
    src_img = ft.Image(width=initial_w, height=initial_w, fit=ft.BoxFit.FILL,
                       border_radius=6, src=_PLACEHOLDER_PNG)
    preview_img = ft.Image(width=initial_w, height=initial_w, fit=ft.BoxFit.FILL,
                           border_radius=6, src=_PREVIEW_PLACEHOLDER_PNG)

    # Shown once every alignable capture is Accepted or Excluded (overview before Finalize).
    overlay_gallery_title = ft.Text(
        "All overlays — confirm before Finalize",
        size=13,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.CYAN_200,
        visible=False,
    )
    overlay_gallery = ft.Row(spacing=10, wrap=True, visible=False)
    _GALLERY_TILE = 128

    layer_hint = (
        f" (image{focus_layer + 1})" if focus_layer is not None else ""
    )
    status_text = ft.Text(
        f"Select a capture on the left to begin{layer_hint}.",
        size=13, color=ft.Colors.CYAN_200,
    )
    cc_text = ft.Text("", size=13, color=ft.Colors.GREY_300)

    ref_caption = ft.Text(
        _ref_label(plans[0]),
        size=13, weight=ft.FontWeight.BOLD)
    src_caption = ft.Text("Source", size=13, weight=ft.FontWeight.BOLD)
    captures_heading = ft.Text(
        f"Captures (aligned to Capture {_ref_idx() + 1})",
        size=14,
        weight=ft.FontWeight.BOLD,
    )

    def cur_points():
        key = (state["visit"], state["capture"])
        if key not in points:
            points[key] = {"ref": [], "src": []}
        return points[key]

    def to_display_pts(full_pts):
        s = state["disp_scale"]
        return [[x / s, y / s] for (x, y) in full_pts]

    def _disp_w() -> int:
        return int(state["disp_w"])

    def _apply_display_width(new_w: int, *, force: bool = False):
        """Resize the three image panels to ``new_w`` and refresh content."""
        new_w = max(DISPLAY_W_MIN, min(DISPLAY_W_MAX, int(new_w)))
        if not force and new_w == state.get("disp_w"):
            return
        state["disp_w"] = new_w
        ref_img.width = new_w
        src_img.width = new_w
        preview_img.width = new_w
        cap = state["capture"]
        if cap is None:
            ref_img.height = new_w
            src_img.height = new_w
            preview_img.height = new_w
            ref_img.src = _placeholder_png("No capture selected", side=new_w)
            src_img.src = _placeholder_png("No capture selected", side=new_w)
            preview_img.src = _placeholder_png("No preview computed", side=new_w)
            try:
                page.update()
            except Exception:
                pass
            return
        render_images()
        plan = plans_by_name[state["visit"]]
        try:
            if cap == _ref_idx():
                stack = display_stack_for(plan)
                preview_img.src = to_png_bytes(stack[_ref_idx()], max_side=new_w)
            else:
                _refresh_preview(plan, cap)
        except Exception:
            pass

    def _on_page_resize(e):
        pw = getattr(e, "width", None) or getattr(page, "width", None)
        _apply_display_width(fit_display_width(pw))
        if _prev_on_resize is not None:
            _prev_on_resize(e)

    def _stored_matrix(plan, cap):
        return overrides[plan.visit_name].get(cap, plan.matrices[cap])

    def _current_matrix(plan, cap):
        """Preview / nudge base: pending → baseline → accepted/auto."""
        if state.get("_pending_matrix") is not None:
            return state["_pending_matrix"]
        if state.get("_baseline_matrix") is not None:
            return state["_baseline_matrix"]
        return _stored_matrix(plan, cap)

    def _set_matrix(matrix, *, acceptible: bool = True):
        """Remember matrix for preview; optionally make it Accept-ready."""
        m = np.asarray(matrix, dtype=np.float32).reshape(2, 3).copy()
        state["_baseline_matrix"] = m
        if acceptible:
            state["_pending_matrix"] = m
        else:
            state.pop("_pending_matrix", None)

    def _sync_src_pins_to_matrix(matrix):
        """Keep ref pin positions; rewrite src pins so they match ``matrix``."""
        pts = cur_points()
        refs = pts["ref"]
        if not refs:
            return
        m = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
        pts["src"] = [
            [
                float(m[0, 0] * x + m[0, 1] * y + m[0, 2]),
                float(m[1, 0] * x + m[1, 1] * y + m[1, 2]),
            ]
            for x, y in refs
        ]

    def _refresh_preview(plan, cap, *, note: str = ""):
        try:
            _show_matrix_preview(plan, cap, _current_matrix(plan, cap), note=note)
        except Exception:
            pass

    def _diag_mask(plan, cap, matrix):
        key = (plan.visit_name, cap)
        d = diag.get(key)
        if not d:
            return None
        resid = correspondence_residuals(d["ref"], d["src"], matrix)
        return resid <= float(state["residual_px"])

    def render_images():
        plan = plans_by_name[state["visit"]]
        cap = state["capture"]
        if cap is None:
            return
        stack = display_stack_for(plan)
        ref_i = _ref_idx()
        ref_full = stack[ref_i]
        src_full = stack[cap]
        dw = _disp_w()
        ref_disp, disp_h, scale = _resize_for_display(ref_full, width=dw)
        src_disp, _, _ = _resize_for_display(src_full, width=dw)
        state["disp_scale"] = scale
        ref_img.width = dw
        src_img.width = dw
        preview_img.width = dw
        ref_img.height = disp_h
        src_img.height = disp_h
        preview_img.height = disp_h

        # Optional diagnostic ORB matches under the pin landmarks.
        if state["show_diag"] and cap != ref_i:
            matrix = _current_matrix(plan, cap)
            mask = _diag_mask(plan, cap, matrix)
            key = (plan.visit_name, cap)
            if mask is not None and key in diag:
                s = state["disp_scale"]
                ref_d = diag[key]["ref"] / s
                src_d = diag[key]["src"] / s
                ref_disp = draw_diagnostic_matches(ref_disp, ref_d, mask)
                src_disp = draw_diagnostic_matches(src_disp, src_d, mask)

        pts = cur_points()
        sel = state.get("selected")
        ref_sel = sel["idx"] if sel and sel["which"] == "ref" else None
        src_sel = sel["idx"] if sel and sel["which"] == "src" else None
        ref_marked = draw_landmarks(
            ref_disp, to_display_pts(pts["ref"]), color=(0, 255, 0), selected_idx=ref_sel
        )
        src_marked = draw_landmarks(
            src_disp, to_display_pts(pts["src"]), color=(255, 0, 255), selected_idx=src_sel
        )
        ref_img.src = to_png_bytes(ref_marked)
        src_img.src = to_png_bytes(src_marked)
        ref_caption.value = f"{_ref_label(plan)} — green"
        src_caption.value = (
            f"Source (Capture {cap+1}{_capture_folder_suffix(plan, cap)}) — magenta"
        )
        page.update()

    def _show_matrix_preview(plan, cap, matrix, *, note: str = ""):
        stack = display_stack_for(plan)
        ref_i = _ref_idx(plan.visit_name)
        overlay = make_overlay(stack[ref_i], stack[cap], matrix)
        preview_img.src = to_png_bytes(overlay, max_side=_disp_w())
        cc = compute_alignment_cc(stack[ref_i], stack[cap], matrix)
        n_in = n_tot = None
        mask = _diag_mask(plan, cap, matrix)
        if mask is not None:
            n_in, n_tot = int(mask.sum()), int(mask.size)
        parts = [f"Score: {cc:.3f}"]
        if n_tot is not None:
            parts.append(f"ORB inliers: {n_in}/{n_tot} (≤{state['residual_px']:.0f}px)")
        if note:
            parts.append(note)
        cc_text.value = "   ".join(parts)
        cc_text.color = ft.Colors.CYAN_200
        page.update()

    def _visit_review_complete(plan) -> bool:
        """
        True when every alignable capture is Accepted (override), Excluded, or
        is the reference — ready for an all-overlays confirmation overview.
        """
        visit = plan.visit_name
        ref_i = _ref_idx(visit)
        for idx in range(len(plan.matrices)):
            if idx == ref_i:
                continue
            if idx in excluded[visit]:
                continue
            if idx in overrides[visit]:
                continue
            return False
        # Need at least one non-excluded capture remaining for averaging.
        n_caps = len(plan.matrices)
        if n_caps - len(excluded[visit]) < 1:
            return False
        # Only show overview after the user has made at least one decision
        # (otherwise a single-capture visit would always look "complete").
        if not overrides[visit] and not excluded[visit]:
            return False
        return True

    def _overlay_png_for_capture(plan, idx, max_side: int = _GALLERY_TILE) -> bytes:
        stack = display_stack_for(plan)
        ref_i = _ref_idx(plan.visit_name)
        if idx == ref_i:
            return to_png_bytes(stack[ref_i], max_side=max_side)
        matrix = overrides[plan.visit_name].get(idx, plan.matrices[idx])
        overlay = make_overlay(stack[ref_i], stack[idx], matrix)
        return to_png_bytes(overlay, max_side=max_side)

    def refresh_overlay_gallery(*, do_update: bool = True):
        """
        When every non-reference capture is Accepted or Excluded, show a strip of
        overlay previews for all non-excluded captures so the user can confirm
        before Finalize.
        """
        plan = plans_by_name[state["visit"]]
        complete = _visit_review_complete(plan)
        overlay_gallery.controls.clear()
        if not complete:
            overlay_gallery.visible = False
            overlay_gallery_title.visible = False
            if do_update:
                try:
                    page.update()
                except Exception:
                    pass
            return

        visit = plan.visit_name
        ref_i = _ref_idx(visit)
        for idx in range(len(plan.matrices)):
            if idx in excluded[visit]:
                continue
            try:
                png = _overlay_png_for_capture(plan, idx)
            except Exception:
                png = _PREVIEW_PLACEHOLDER_PNG
            is_ref = idx == ref_i
            is_corr = idx in overrides[visit]
            if is_ref:
                badge, col = "ref", ft.Colors.CYAN_300
            elif is_corr:
                badge, col = "accepted", ft.Colors.GREEN_400
            else:
                badge, col = "auto", ft.Colors.GREY_400
            label = f"Cap {idx + 1}"
            if is_ref:
                label += " (ref)"
            selected = idx == state["capture"]
            overlay_gallery.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Image(
                                src=png,
                                width=_GALLERY_TILE,
                                height=_GALLERY_TILE,
                                fit=ft.BoxFit.CONTAIN,
                                border_radius=4,
                            ),
                            ft.Text(label, size=11, weight=ft.FontWeight.BOLD),
                            ft.Text(badge, size=10, color=col),
                        ],
                        spacing=2,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        tight=True,
                    ),
                    padding=6,
                    border_radius=6,
                    bgcolor=ft.Colors.CYAN_900 if selected else ft.Colors.GREY_900,
                    border=ft.Border.all(
                        1, ft.Colors.CYAN_400 if selected else ft.Colors.GREY_700
                    ),
                    on_click=lambda _, i=idx: select_capture(i),
                    tooltip=f"Show Capture {idx + 1} in the editor",
                )
            )

        overlay_gallery.visible = True
        overlay_gallery_title.visible = True
        overlay_gallery_title.value = (
            f"All overlays ({len(overlay_gallery.controls)} captures) — "
            "confirm before Finalize"
        )
        if do_update:
            try:
                page.update()
            except Exception:
                pass

    def _load_diagnostics(plan, idx):
        key = (plan.visit_name, idx)
        if key in diag:
            return
        stack = display_stack_for(plan)
        ref_i = _ref_idx(plan.visit_name)
        extracted = extract_feature_correspondences(stack[ref_i], stack[idx])
        if extracted is None:
            diag[key] = None
            return
        ref_pts, src_pts = extracted
        diag[key] = {"ref": ref_pts, "src": src_pts}

    def _sync_ref_dropdown(plan):
        n = len(plan.matrices)
        ref_dropdown.options = [
            ft.DropdownOption(
                key=str(i),
                text=f"Capture {i + 1}{_capture_folder_suffix(plan, i)}",
            )
            for i in range(n)
        ]
        ref_dropdown.value = str(_ref_idx(plan.visit_name))
        captures_heading.value = (
            f"Captures (aligned to Capture {_ref_idx(plan.visit_name) + 1})"
        )

    def refresh_capture_list(*, do_update: bool = True):
        plan = plans_by_name[state["visit"]]
        capture_list.controls.clear()
        ref_i = _ref_idx(plan.visit_name)
        for idx in range(0, len(plan.matrices)):
            is_excl = idx in excluded[plan.visit_name]
            corrected = idx in overrides[plan.visit_name]
            is_ref = idx == ref_i
            if is_excl:
                badge, col = "excluded", ft.Colors.RED_400
                icon = ft.Icons.BLOCK
            elif is_ref:
                badge, col = "reference", ft.Colors.CYAN_300
                icon = ft.Icons.PUSH_PIN
            elif corrected:
                badge, col = "corrected", ft.Colors.GREEN_400
                icon = ft.Icons.CHECK_CIRCLE
            else:
                badge, col = "auto", ft.Colors.GREY_400
                icon = ft.Icons.CIRCLE_OUTLINED
            selected = idx == state["capture"]
            label = f"Capture {idx+1}"
            if is_ref:
                label += " (ref)"
            label += _capture_folder_suffix(plan, idx)
            capture_list.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Icon(icon, color=col, size=16),
                        ft.Text(label, size=13,
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
        if do_update:
            try:
                page.update()
            except Exception:
                pass

    def _seed_auto_points(plan, idx, *, force: bool = False):
        """Prefill editable landmarks from the current auto / accepted matrix."""
        key = (plan.visit_name, idx)
        if not force and key in points and (points[key]["ref"] or points[key]["src"]):
            return
        stack = display_stack_for(plan)
        ref_i = _ref_idx(plan.visit_name)
        h, w = stack[ref_i].shape[:2]
        matrix = overrides[plan.visit_name].get(idx, plan.matrices[idx])
        ref_pts, src_pts = seed_correspondences_from_matrix(matrix, h, w, n_points=6)
        points[key] = {"ref": ref_pts, "src": src_pts}
        tx, ty = float(matrix[0, 2]), float(matrix[1, 2])
        status_text.value = (
            f"Capture {idx+1}: auto shift ≈ ({tx:.1f}, {ty:.1f}) px — 6 editable pins "
            f"+ ORB diagnostics. Nudge overlay, drop outliers, or edit pins → Compute."
        )

    def select_capture(idx):
        state["capture"] = idx
        preview_img.src = _PREVIEW_PLACEHOLDER_PNG
        cc_text.value = ""
        plan = plans_by_name[state["visit"]]
        state.pop("_pending_matrix", None)
        state["_baseline_matrix"] = None
        ref_i = _ref_idx()

        # Reference capture — point editing N/A; may still be excluded from average.
        if idx == ref_i:
            if idx in excluded[state["visit"]]:
                status_text.value = (
                    f"Capture {idx + 1} (reference) is EXCLUDED from the average. "
                    "Toggle Exclude off to include it again."
                )
            else:
                status_text.value = (
                    f"Capture {idx + 1} is the alignment reference (identity). "
                    "Point correction is not needed; use Exclude if this capture "
                    "should be omitted from the average."
                )
            try:
                stack = display_stack_for(plan)
                # Show reference alone (no warp overlay).
                preview_img.src = to_png_bytes(stack[ref_i], max_side=_disp_w())
                cc_text.value = "Reference capture — Exclude available"
                cc_text.color = ft.Colors.GREY_300
            except Exception:
                pass
            refresh_capture_list()
            render_images()
            return

        _seed_auto_points(plan, idx, force=False)
        try:
            _load_diagnostics(plan, idx)
        except Exception:
            pass
        n_ref = len(cur_points()["ref"])
        n_src = len(cur_points()["src"])
        if idx in excluded[state["visit"]]:
            status_text.value = (
                f"Capture {idx+1} is EXCLUDED from the average. "
                f"Toggle Exclude off to include it again."
            )
        elif not status_text.value.startswith(f"Capture {idx+1}: auto shift"):
            status_text.value = (
                f"Capture {idx+1}: {n_ref}/{n_src} pins + ORB diagnostics. "
                f"Nudge, drop outliers, or edit pins → Compute & Preview."
            )
        try:
            matrix = _stored_matrix(plan, idx)
            # Baseline is Accept-ready so auto alignment can be confirmed as-is.
            _set_matrix(matrix, acceptible=True)
            _show_matrix_preview(plan, idx, matrix, note="baseline")
        except Exception:
            pass
        refresh_capture_list()
        render_images()

    def change_reference(new_ref: int):
        """
        Re-run automatic registration onto ``new_ref`` for the focus image only,
        then refresh the manual corresponding-point editor.
        """
        visit = state["visit"]
        plan = plans_by_name[visit]
        new_ref = int(new_ref)
        old_ref = _ref_idx(visit)
        if new_ref == old_ref:
            return
        if not (0 <= new_ref < len(plan.matrices)):
            return

        layer_label = (
            f"image{focus_layer + 1}" if focus_layer is not None else "Image 5"
        )
        status_text.value = (
            f"Auto-registering → Capture {new_ref + 1} "
            f"({layer_label} of this Visit only)…"
        )
        cc_text.value = ""
        ref_dropdown.disabled = True
        try:
            page.update()
        except Exception:
            pass

        stack = display_stack_for(plan)
        logs: list[str] = []

        def _log(msg: str):
            logs.append(str(msg))

        def _apply_success():
            overrides[visit].clear()
            ref_by_visit[visit] = int(plan.reference_idx)
            for key in list(points.keys()):
                if key[0] == visit:
                    points.pop(key, None)
            for key in list(diag.keys()):
                if key[0] == visit:
                    diag.pop(key, None)
            state.pop("_pending_matrix", None)
            state["_baseline_matrix"] = None
            state["selected"] = None
            state["dragging"] = False
            _sync_ref_dropdown(plan)
            captures_heading.value = (
                f"Captures (aligned to Capture {new_ref + 1})"
            )
            status_text.value = (
                f"Auto-registered to Capture {new_ref + 1} ({layer_label}). "
                "Pins cleared — select a capture to refine corresponding points."
            )
            cap = state["capture"]
            refresh_capture_list(do_update=False)
            refresh_overlay_gallery(do_update=False)
            if cap is None:
                ref_caption.value = _ref_label(plan)
                src_caption.value = "Source"
            else:
                # Avoid nested page.update from select_capture until enabled again.
                select_capture(cap)

        def _run_realign():
            realign_plan_to_reference(
                plan,
                new_ref,
                stack=stack,
                layer_idx=None,  # stack already scoped to focus layer
                auto_refine=True,
                log_callback=_log,
            )

        def _finish_ok():
            ref_dropdown.disabled = False
            _apply_success()
            try:
                page.update()
            except Exception:
                pass

        def _finish_err(exc: BaseException):
            ref_dropdown.disabled = False
            ref_dropdown.value = str(old_ref)
            status_text.value = (
                f"Auto-registration to Capture {new_ref + 1} failed: {exc}"
            )
            cc_text.value = ""
            try:
                page.update()
            except Exception:
                pass

        run_thread = getattr(page, "run_thread", None)
        if callable(run_thread):
            def work():
                try:
                    _run_realign()
                    run_task = getattr(page, "run_task", None)
                    if callable(run_task):
                        async def _ui():
                            _finish_ok()
                        run_task(_ui)
                    else:
                        _finish_ok()
                except Exception as exc:
                    run_task = getattr(page, "run_task", None)
                    if callable(run_task):
                        async def _ui_err(e=exc):
                            _finish_err(e)
                        run_task(_ui_err)
                    else:
                        _finish_err(exc)

            run_thread(work)
        else:
            # Headless / smoke tests: run synchronously.
            try:
                _run_realign()
                _finish_ok()
            except Exception as exc:
                _finish_err(exc)

    def on_ref_dropdown_select(e):
        raw = getattr(ref_dropdown, "value", None)
        if raw is None and e is not None:
            raw = getattr(getattr(e, "control", None), "value", None)
        if raw is None:
            return
        try:
            change_reference(int(raw))
        except (TypeError, ValueError):
            pass

    def select_visit(name):
        state["visit"] = name
        state["capture"] = None
        state.pop("_pending_matrix", None)
        state["_baseline_matrix"] = None
        plan = plans_by_name[name]
        _sync_ref_dropdown(plan)
        refresh_capture_list()
        refresh_overlay_gallery(do_update=False)
        dw = _disp_w()
        ref_img.width = dw
        src_img.width = dw
        preview_img.width = dw
        ref_img.height = dw
        src_img.height = dw
        preview_img.height = dw
        ref_img.src = _placeholder_png("No capture selected", side=dw)
        src_img.src = _placeholder_png("No capture selected", side=dw)
        preview_img.src = _placeholder_png("No preview computed", side=dw)
        ref_caption.value = _ref_label(plan)
        src_caption.value = "Source"
        status_text.value = f"Select a capture on the left to begin{layer_hint}."
        page.update()

    def _event_xy(e):
        pos = getattr(e, "local_position", None)
        if pos is None:
            return None
        return float(pos.x), float(pos.y)

    def _hit_pin(which, disp_x, disp_y):
        disp_pts = to_display_pts(cur_points()[which])
        return nearest_landmark_index(disp_pts, disp_x, disp_y, _PIN_HIT_PX)

    def add_point(which, e):
        if state["capture"] is None or state["capture"] == _ref_idx():
            return
        if state.get("suppress_add"):
            state["suppress_add"] = False
            return
        xy = _event_xy(e)
        if xy is None:
            return
        # Tap on an existing pin → select it (drag handles the move).
        hit = _hit_pin(which, xy[0], xy[1])
        if hit is not None:
            state["selected"] = {"which": which, "idx": hit}
            status_text.value = (
                f"Selected {which} pin #{hit + 1}. Drag to move, or Delete selected."
            )
            render_images()
            return
        s = state["disp_scale"]
        fx, fy = xy[0] * s, xy[1] * s
        cur_points()[which].append([fx, fy])
        state["selected"] = {"which": which, "idx": len(cur_points()[which]) - 1}
        # Keep nudged/baseline overlay; Accept requires Compute after pin edits.
        state.pop("_pending_matrix", None)
        n_ref = len(cur_points()["ref"])
        n_src = len(cur_points()["src"])
        status_text.value = (
            f"Points: ref {n_ref}, src {n_src}. "
            + ("Ready — Compute & Preview." if (n_ref == n_src and n_ref >= 3)
               else "Add matching pairs (≥3, equal counts). Overlay keeps last nudge.")
        )
        plan = plans_by_name[state["visit"]]
        _refresh_preview(plan, state["capture"], note="pins editing")
        render_images()

    def pan_start(which, e):
        if state["capture"] is None or state["capture"] == _ref_idx():
            return
        xy = _event_xy(e)
        if xy is None:
            return
        hit = _hit_pin(which, xy[0], xy[1])
        if hit is None:
            state["dragging"] = False
            return
        state["selected"] = {"which": which, "idx": hit}
        state["dragging"] = True
        state["suppress_add"] = True
        status_text.value = f"Dragging {which} pin #{hit + 1}…"
        render_images()

    def pan_update(which, e):
        if not state.get("dragging"):
            return
        sel = state.get("selected")
        if not sel or sel["which"] != which:
            return
        xy = _event_xy(e)
        if xy is None:
            return
        s = state["disp_scale"]
        pts = cur_points()[which]
        idx = sel["idx"]
        if idx < 0 or idx >= len(pts):
            return
        pts[idx] = [xy[0] * s, xy[1] * s]
        state.pop("_pending_matrix", None)
        render_images()

    def pan_end(which, e):
        if not state.get("dragging"):
            return
        state["dragging"] = False
        sel = state.get("selected")
        if sel and sel["which"] == which:
            status_text.value = (
                f"Moved {which} pin #{sel['idx'] + 1}. "
                f"Overlay keeps last nudge — Compute & Preview to refit."
            )
        state["suppress_add"] = True
        plan = plans_by_name[state["visit"]]
        _refresh_preview(plan, state["capture"], note="pins editing")
        render_images()

    def undo_point(which, e):
        if state["capture"] is None:
            return
        pts = cur_points()[which]
        if pts:
            pts.pop()
        state["selected"] = None
        state.pop("_pending_matrix", None)
        plan = plans_by_name[state["visit"]]
        _refresh_preview(plan, state["capture"], note="pins editing")
        render_images()

    def delete_selected(e):
        if state["capture"] is None:
            return
        sel = state.get("selected")
        if not sel:
            status_text.value = "No pin selected. Click or drag a numbered pin first."
            page.update()
            return
        which, idx = sel["which"], sel["idx"]
        pts = cur_points()
        # Prefer deleting the paired index on both sides when counts match.
        n_ref, n_src = len(pts["ref"]), len(pts["src"])
        if n_ref == n_src and 0 <= idx < n_ref:
            pts["ref"].pop(idx)
            pts["src"].pop(idx)
            status_text.value = f"Deleted pair #{idx + 1} from both images."
        elif 0 <= idx < len(pts[which]):
            pts[which].pop(idx)
            status_text.value = f"Deleted {which} pin #{idx + 1}."
        else:
            status_text.value = "Selected pin no longer exists."
        state["selected"] = None
        state.pop("_pending_matrix", None)
        render_images()

    def reset_points(e):
        if state["capture"] is None or state["capture"] == _ref_idx():
            return
        plan = plans_by_name[state["visit"]]
        _seed_auto_points(plan, state["capture"], force=True)
        state["selected"] = None
        matrix = _stored_matrix(plan, state["capture"])
        _set_matrix(matrix, acceptible=False)
        status_text.value = (
            "Pins reset to automatic landmarks. Drag a numbered pin to adjust."
        )
        try:
            _show_matrix_preview(plan, state["capture"], matrix, note="reset")
        except Exception:
            preview_img.src = _PREVIEW_PLACEHOLDER_PNG
            cc_text.value = ""
        render_images()

    def clear_points(e):
        if state["capture"] is None:
            return
        if state["capture"] == _ref_idx():
            status_text.value = (
                f"Capture {_ref_idx() + 1} (ref) has no editable pins."
            )
            page.update()
            return
        key = (state["visit"], state["capture"])
        points[key] = {"ref": [], "src": []}
        state["selected"] = None
        # Keep nudged / baseline transform so user can place fresh points after nudge.
        plan = plans_by_name[state["visit"]]
        cap = state["capture"]
        if state.get("_pending_matrix") is None and state.get("_baseline_matrix") is not None:
            # Pins wiped after edits — nudge/baseline still Accept-able until new pins added.
            state["_pending_matrix"] = np.asarray(
                state["_baseline_matrix"], dtype=np.float32
            ).copy()
        status_text.value = (
            "Pins cleared — overlay nudge kept. Now click matching landmarks "
            "(ref green → source magenta), ≥3 pairs, then Compute & Preview."
        )
        _refresh_preview(plan, cap, note="ready for new pins")
        render_images()

    def toggle_exclude(e):
        if state["capture"] is None:
            return
        visit = state["visit"]
        idx = state["capture"]
        plan = plans_by_name[visit]
        n_caps = len(plan.matrices)
        if idx in excluded[visit]:
            excluded[visit].discard(idx)
            status_text.value = f"Capture {idx+1} included in the average again."
        else:
            # Keep at least one capture in the average.
            remaining = n_caps - len(excluded[visit]) - 1
            if remaining < 1:
                status_text.value = (
                    "Cannot exclude every capture — at least one must remain for averaging."
                )
                page.update()
                return
            excluded[visit].add(idx)
            overrides[visit].pop(idx, None)
            state.pop("_pending_matrix", None)
            status_text.value = (
                f"Capture {idx+1} EXCLUDED from average (poor quality / unalignable)."
            )
        refresh_capture_list()
        refresh_overlay_gallery()
        page.update()

    def compute_preview(e):
        if state["capture"] is None:
            return
        if state["capture"] == _ref_idx():
            status_text.value = (
                f"Capture {_ref_idx() + 1} is the reference — "
                "use Exclude instead of point correction."
            )
            page.update()
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

        stack = display_stack_for(plan)
        ref_i = _ref_idx()
        auto_cc = compute_alignment_cc(stack[ref_i], stack[cap], plan.matrices[cap])
        manual_cc = compute_alignment_cc(stack[ref_i], stack[cap], matrix)
        _set_matrix(matrix, acceptible=True)
        note = (
            f"pins → manual {manual_cc:.3f} vs auto {auto_cc:.3f}"
            + (" (better ✓)" if manual_cc >= auto_cc else " (worse — check pins)")
        )
        _show_matrix_preview(plan, cap, matrix, note=note)
        cc_text.color = ft.Colors.GREEN_400 if manual_cc >= auto_cc else ft.Colors.AMBER_400
        status_text.value = "Preview ready. Green = reference, magenta = source; white = aligned."
        render_images()
        page.update()

    def accept_correction(e):
        if state["capture"] == _ref_idx():
            status_text.value = (
                f"Capture {_ref_idx() + 1} is the reference — "
                "use Exclude instead of Accept."
            )
            page.update()
            return
        # Pending (nudge/compute) preferred; else baseline so auto can be Accepted as-is.
        mat = state.get("_pending_matrix")
        if mat is None:
            mat = state.get("_baseline_matrix")
        if mat is None or state["capture"] is None:
            status_text.value = (
                "Select a capture, then Nudge / Compute & Preview, or Accept the "
                "auto baseline."
            )
            page.update()
            return
        visit = state["visit"]
        idx = state["capture"]
        overrides[visit][idx] = np.asarray(mat, dtype=np.float32).reshape(2, 3).copy()
        excluded[visit].discard(idx)
        status_text.value = f"Capture {idx+1} correction accepted."
        refresh_capture_list()
        refresh_overlay_gallery()
        page.update()

    def revert_auto(e):
        if state["capture"] is None or state["capture"] == _ref_idx():
            return
        visit = state["visit"]
        idx = state["capture"]
        overrides[visit].pop(idx, None)
        excluded[visit].discard(idx)
        plan = plans_by_name[visit]
        _seed_auto_points(plan, idx, force=True)
        _set_matrix(plan.matrices[idx], acceptible=False)
        status_text.value = f"Capture {idx+1} reverted to automatic alignment landmarks."
        try:
            _show_matrix_preview(plan, idx, plan.matrices[idx], note="auto")
        except Exception:
            preview_img.src = _PREVIEW_PLACEHOLDER_PNG
            cc_text.value = ""
        refresh_capture_list()
        refresh_overlay_gallery()
        render_images()

    def apply_nudge(dx=0.0, dy=0.0, dtheta=0.0):
        if state["capture"] is None or state["capture"] == _ref_idx():
            return
        plan = plans_by_name[state["visit"]]
        cap = state["capture"]
        stack = display_stack_for(plan)
        ref_i = _ref_idx()
        h, w = stack[ref_i].shape[:2]
        base = _current_matrix(plan, cap)
        matrix = nudge_affine_matrix(
            base, dx=dx, dy=dy, dtheta_deg=dtheta, center=(w / 2.0, h / 2.0)
        )
        _set_matrix(matrix, acceptible=True)
        pts = cur_points()
        if pts["ref"]:
            # Preserve user-chosen ref landmarks; slide src onto the new transform.
            if len(pts["ref"]) == len(pts["src"]) or not pts["src"]:
                _sync_src_pins_to_matrix(matrix)
            else:
                ref_pts, src_pts = seed_correspondences_from_matrix(
                    matrix, h, w, n_points=max(3, min(8, len(pts["ref"])))
                )
                points[(plan.visit_name, cap)] = {"ref": ref_pts, "src": src_pts}
            status_text.value = (
                f"Nudged (Δx={dx:+.0f}, Δy={dy:+.0f}, Δθ={dtheta:+.1f}°). "
                f"Drag pins to refine, or Clear points then click new pairs — "
                f"then Compute & Preview (or Accept nudge as-is)."
            )
        else:
            # Pins already cleared: keep empty so user can specify points next.
            status_text.value = (
                f"Nudged (Δx={dx:+.0f}, Δy={dy:+.0f}, Δθ={dtheta:+.1f}°). "
                f"Now click matching landmarks (≥3 pairs), then Compute & Preview — "
                f"or Accept this nudge without pins."
            )
        _show_matrix_preview(plan, cap, matrix, note="nudge")
        render_images()

    def drop_outliers_refit(e):
        if state["capture"] is None or state["capture"] == _ref_idx():
            return
        plan = plans_by_name[state["visit"]]
        cap = state["capture"]
        key = (plan.visit_name, cap)
        d = diag.get(key)
        if not d:
            status_text.value = "No ORB diagnostics for this capture — cannot drop outliers."
            page.update()
            return
        matrix = _current_matrix(plan, cap)
        ref_in, src_in, keep = filter_correspondences_by_residual(
            d["ref"], d["src"], matrix, state["residual_px"]
        )
        if ref_in.shape[0] < 3:
            status_text.value = (
                f"Only {ref_in.shape[0]} inliers at ≤{state['residual_px']:.0f}px — "
                f"relax the residual slider or nudge first."
            )
            page.update()
            return
        try:
            new_m = estimate_affine_from_correspondences(ref_in, src_in)
        except ValueError as ex:
            status_text.value = f"Refit failed: {ex}"
            page.update()
            return
        _set_matrix(new_m, acceptible=True)
        pts = cur_points()
        if pts["ref"] and len(pts["ref"]) == len(pts["src"]):
            _sync_src_pins_to_matrix(new_m)
        else:
            stack = display_stack_for(plan)
            ref_i = _ref_idx()
            h, w = stack[ref_i].shape[:2]
            ref_pts, src_pts = seed_correspondences_from_matrix(new_m, h, w, n_points=6)
            points[key] = {"ref": ref_pts, "src": src_pts}
        n_drop = int((~keep).sum())
        status_text.value = (
            f"Dropped {n_drop} outliers, refit from {ref_in.shape[0]} inliers. "
            f"Adjust pins further or Accept."
        )
        _show_matrix_preview(plan, cap, new_m, note="outlier refit")
        render_images()

    def on_residual_change(e):
        state["residual_px"] = float(e.control.value)
        residual_label.value = f"Residual ≤ {state['residual_px']:.0f} px"
        if state["capture"] is not None and state["capture"] != _ref_idx():
            plan = plans_by_name[state["visit"]]
            cap = state["capture"]
            try:
                _show_matrix_preview(plan, cap, _current_matrix(plan, cap))
            except Exception:
                pass
            render_images()
        else:
            page.update()

    def on_show_diag_change(e):
        state["show_diag"] = bool(e.control.value)
        render_images()

    def finalize_click(e):
        clean_overrides = {v: dict(d) for v, d in overrides.items() if d}
        clean_excluded = {v: sorted(s) for v, s in excluded.items() if s}
        clean_points = {}
        for (visit, cap), pt in points.items():
            if visit in clean_overrides and cap in clean_overrides[visit]:
                clean_points.setdefault(visit, {})[cap] = pt
        ref_changed = any(
            ref_by_visit[v] != initial_ref_by_visit.get(v, 0)
            for v in ref_by_visit
        )
        if not clean_overrides and not clean_excluded and not ref_changed:
            status_text.value = (
                "No corrections, exclusions, or reference change yet. "
                "Accept a preview, Exclude a capture, or change the reference."
            )
            page.update()
            return
        # Prefer 3-arg callback (overrides, points, excluded); fall back for older hooks.
        try:
            on_finalize(clean_overrides, clean_points, clean_excluded)
        except TypeError:
            on_finalize(clean_overrides, clean_points)

    ref_dropdown.on_select = on_ref_dropdown_select

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
        drag_interval=33,
        mouse_cursor=ft.MouseCursor.MOVE,
        on_tap_down=lambda e: add_point("ref", e),
        on_pan_start=lambda e: pan_start("ref", e),
        on_pan_update=lambda e: pan_update("ref", e),
        on_pan_end=lambda e: pan_end("ref", e),
    )
    src_gd = ft.GestureDetector(
        content=src_img,
        drag_interval=33,
        mouse_cursor=ft.MouseCursor.MOVE,
        on_tap_down=lambda e: add_point("src", e),
        on_pan_start=lambda e: pan_start("src", e),
        on_pan_update=lambda e: pan_update("src", e),
        on_pan_end=lambda e: pan_end("src", e),
    )

    step = state["nudge_step"]
    residual_label = ft.Text(
        f"Residual ≤ {state['residual_px']:.0f} px", size=12, color=ft.Colors.GREY_300
    )
    residual_slider = ft.Slider(
        min=2, max=20, divisions=18, value=state["residual_px"],
        label="{value} px", width=180, on_change=on_residual_change,
    )
    show_diag_switch = ft.Switch(
        label="Show ORB diagnostics",
        value=state["show_diag"],
        on_change=on_show_diag_change,
    )

    nudge_row = ft.Row([
        ft.Text("Nudge overlay", size=12, color=ft.Colors.GREY_300),
        ft.OutlinedButton("←", on_click=lambda e: apply_nudge(dx=-step)),
        ft.OutlinedButton("→", on_click=lambda e: apply_nudge(dx=+step)),
        ft.OutlinedButton("↑", on_click=lambda e: apply_nudge(dy=-step)),
        ft.OutlinedButton("↓", on_click=lambda e: apply_nudge(dy=+step)),
        ft.OutlinedButton("↺", tooltip="Rotate CCW 0.5°",
                          on_click=lambda e: apply_nudge(dtheta=+0.5)),
        ft.OutlinedButton("↻", tooltip="Rotate CW 0.5°",
                          on_click=lambda e: apply_nudge(dtheta=-0.5)),
        ft.Text(f"step {step:.0f}px / 0.5°", size=11, color=ft.Colors.GREY_500),
    ], wrap=True, spacing=6)

    diag_row = ft.Row([
        show_diag_switch,
        residual_label,
        residual_slider,
        ft.FilledTonalButton(
            "Drop outliers & refit",
            icon=ft.Icons.FILTER_ALT,
            on_click=drop_outliers_refit,
        ),
    ], wrap=True, spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    control_row = ft.Row([
        ft.OutlinedButton("Undo ref", icon=ft.Icons.UNDO,
                          on_click=lambda e: undo_point("ref", e)),
        ft.OutlinedButton("Undo src", icon=ft.Icons.UNDO,
                          on_click=lambda e: undo_point("src", e)),
        ft.OutlinedButton(
            "Delete selected",
            icon=ft.Icons.DELETE_OUTLINE,
            on_click=delete_selected,
        ),
        ft.OutlinedButton("Reset to auto points", icon=ft.Icons.RESTART_ALT, on_click=reset_points),
        ft.FilledTonalButton(
            "Clear points",
            icon=ft.Icons.CLEAR_ALL,
            on_click=clear_points,
            style=ft.ButtonStyle(color=ft.Colors.AMBER_300),
        ),
        ft.FilledButton("Compute & Preview", icon=ft.Icons.CALCULATE, on_click=compute_preview),
        ft.FilledButton("Accept", icon=ft.Icons.CHECK, on_click=accept_correction),
        ft.OutlinedButton("Revert to auto", icon=ft.Icons.RESTORE, on_click=revert_auto),
        ft.OutlinedButton(
            "Exclude / Include",
            icon=ft.Icons.BLOCK,
            on_click=toggle_exclude,
            style=ft.ButtonStyle(color=ft.Colors.RED_300),
        ),
    ], wrap=True, spacing=8)

    title_suffix = f" — image{focus_layer + 1}" if focus_layer is not None else ""

    left_panel = ft.Container(
        content=ft.Column([
            ft.Text("Visits", size=14, weight=ft.FontWeight.BOLD),
            visit_row,
            ft.Divider(height=10),
            captures_heading,
            ref_dropdown,
            capture_list,
        ], spacing=8, expand=True),
        width=LEFT_PANEL_W,
        padding=8,
    )

    images_row = ft.Row([
        ft.Column([ref_caption, ft.Container(content=ref_gd,
                   border=ft.Border.all(1, ft.Colors.GREEN_700), border_radius=6)]),
        ft.Column([src_caption, ft.Container(content=src_gd,
                   border=ft.Border.all(1, ft.Colors.PURPLE_400), border_radius=6)]),
        ft.Column([ft.Text("Overlay preview", size=13, weight=ft.FontWeight.BOLD),
                   ft.Container(content=preview_img,
                   border=ft.Border.all(1, ft.Colors.CYAN_700), border_radius=6)]),
    ], spacing=16)

    editor_panel = ft.Column([
        images_row,
        overlay_gallery_title,
        overlay_gallery,
        nudge_row,
        diag_row,
        control_row,
        status_text,
        cc_text,
    ], spacing=12, expand=True, scroll=ft.ScrollMode.AUTO)

    def _go_back(e=None):
        page.on_resize = _prev_on_resize
        on_back()

    header = ft.Row([
        ft.IconButton(icon=ft.Icons.ARROW_BACK, tooltip="Back",
                      on_click=_go_back),
        ft.Text(f"Manual Corresponding-Point Correction{title_suffix}", size=22,
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

    page.on_resize = _on_page_resize

    # Initialize selection without page.update — view is not mounted yet.
    _sync_ref_dropdown(plans_by_name[state["visit"]])
    refresh_capture_list(do_update=False)
    return view
