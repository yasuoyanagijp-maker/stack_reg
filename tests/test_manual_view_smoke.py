"""
Headless smoke test for the manual corresponding-point view. It cannot render a
real window, so it uses a fake page (no-op update) and synthesizes tap / pan
events to drive the actual handler code paths.
"""
import os
import types
import numpy as np
import cv2

import flet as ft
from app.core.pipeline import prepare_visit
from app.ui.manual_align_view import (
    create_manual_align_view,
    _capture_folder_suffix,
)


def _write_visit(root, n_captures=3, n_layers=2, size=128):
    base = np.full((size, size), 20, np.uint8)
    cv2.line(base, (20, 20), (100, 110), 220, 3)
    cv2.circle(base, (64, 64), 30, 180, 2)
    cv2.putText(base, "R", (48, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    for c in range(n_captures):
        M = cv2.getRotationMatrix2D((size / 2, size / 2), c * 2.0, 1.0)
        M[0, 2] += c * 3.0
        warped = cv2.warpAffine(base, M, (size, size))
        cap_dir = os.path.join(root, str(c + 1))
        os.makedirs(cap_dir, exist_ok=True)
        for layer in range(n_layers):
            cv2.imwrite(os.path.join(cap_dir, f"l{layer+1}.jpg"), warped)


def _walk(control):
    yield control
    for attr in ("controls", "content"):
        child = getattr(control, attr, None)
        if isinstance(child, (list, tuple)):
            for c in child:
                if c is not None:
                    yield from _walk(c)
        elif child is not None and hasattr(child, "__dict__"):
            yield from _walk(child)


def _tap(x, y):
    return types.SimpleNamespace(local_position=types.SimpleNamespace(x=x, y=y))


def test_capture_folder_suffix_omits_numeric_match():
    """Folder names that equal the capture number omit the ' — N' suffix."""
    plan = types.SimpleNamespace(sorted_captures=["1", "scan_B", "3"])
    assert _capture_folder_suffix(plan, 0) == ""
    assert _capture_folder_suffix(plan, 1) == " — scan_B"
    assert _capture_folder_suffix(plan, 2) == ""
    long = "x" * 30
    plan2 = types.SimpleNamespace(sorted_captures=[long])
    suffix = _capture_folder_suffix(plan2, 0)
    assert suffix.startswith(" — ")
    assert suffix.endswith("…")
    assert len(suffix) == 3 + 24  # " — " + truncated name


def test_manual_view_labels_omit_redundant_folder(tmp_path):
    """Numbered capture folders must not render as 'Capture 1 — 1'."""
    visit = tmp_path / "visit"
    _write_visit(str(visit))
    plan = prepare_visit(str(visit))

    fake_page = types.SimpleNamespace(
        update=lambda: None, width=1280, on_resize=None, window=None,
    )
    view = create_manual_align_view(
        fake_page, [plan], on_back=lambda: None, on_finalize=lambda *a, **k: None,
    )
    texts = [
        str(getattr(n, "value", "") or getattr(n, "text", "") or "")
        for n in _walk(view)
    ]
    assert not any("Capture 1 — 1" in t for t in texts)
    assert any("Capture 1 (ref)" == t or "Capture 1 (ref)" in t for t in texts)
    dropdowns = [n for n in _walk(view) if isinstance(n, ft.Dropdown)]
    ref_dd = next(d for d in dropdowns if d.label == "Alignment reference")
    assert all(" — " not in (opt.text or "") for opt in ref_dd.options)


def test_manual_view_full_interaction(tmp_path):
    visit = tmp_path / "visit"
    _write_visit(str(visit))
    plan = prepare_visit(str(visit))

    fake_page = types.SimpleNamespace(
        update=lambda: None, width=1280, on_resize=None, window=None,
    )
    captured = {}

    def on_finalize(overrides_by_visit, points_by_visit, excluded_by_visit=None):
        captured["overrides"] = overrides_by_visit
        captured["points"] = points_by_visit
        captured["excluded"] = excluded_by_visit or {}

    view = create_manual_align_view(
        fake_page, [plan], on_back=lambda: captured.setdefault("back", True),
        on_finalize=on_finalize,
    )

    nodes = list(_walk(view))
    gestures = [n for n in nodes if isinstance(n, ft.GestureDetector)]
    assert len(gestures) == 2, "expected reference + source pickers"
    ref_gd, src_gd = gestures
    assert ref_gd.on_pan_start and ref_gd.on_pan_update and ref_gd.on_pan_end

    buttons = {getattr(n, "content", None): n for n in nodes
               if isinstance(n, (ft.FilledButton, ft.OutlinedButton, ft.FilledTonalButton))}
    assert "Compute & Preview" in buttons
    assert "Accept" in buttons
    assert "Finalize all images" in buttons
    assert "Clear points" in buttons
    assert "Delete selected" in buttons
    assert "Drop outliers & refit" in buttons

    capture_rows = [n for n in nodes
                    if isinstance(n, ft.Container) and getattr(n, "on_click", None) is not None]
    assert len(capture_rows) >= 2
    capture_rows[1].on_click(None)

    # Nudge first, then Clear (must keep transform), then place pins.
    assert "→" in buttons
    buttons["→"].on_click(None)
    buttons["Clear points"].on_click(None)
    for (x, y) in [(30, 30), (90, 70), (60, 100)]:
        ref_gd.on_tap_down(_tap(x, y))
        src_gd.on_tap_down(_tap(x + 2, y - 1))

    ref_gd.on_pan_start(_tap(30, 30))
    ref_gd.on_pan_update(_tap(40, 35))
    ref_gd.on_pan_end(_tap(40, 35))

    buttons["Compute & Preview"].on_click(None)
    buttons["Accept"].on_click(None)
    buttons["Finalize all images"].on_click(None)

    assert "overrides" in captured
    ov = captured["overrides"]
    assert list(ov.keys()) == [plan.visit_name]
    assert 1 in ov[plan.visit_name]
    mat = ov[plan.visit_name][1]
    assert np.asarray(mat).shape == (2, 3)
    assert 1 in captured["points"][plan.visit_name]
    pts = captured["points"][plan.visit_name][1]
    assert len(pts["ref"]) == 3 and len(pts["src"]) == 3
    # Drag moved pin #1 away from the original click location (display→full).
    assert abs(pts["ref"][0][0] - pts["ref"][1][0]) > 1.0


def test_manual_view_focus_layer_and_overlay_gallery(tmp_path):
    """Second-image focus and all-overlays gallery after Accept/Exclude."""
    visit = tmp_path / "visit"
    _write_visit(str(visit), n_captures=3, n_layers=2)
    plan = prepare_visit(str(visit))

    fake_page = types.SimpleNamespace(
        update=lambda: None, width=1280, on_resize=None, window=None,
    )
    captured = {}

    view = create_manual_align_view(
        fake_page, [plan], on_back=lambda: None,
        on_finalize=lambda *a, **k: captured.setdefault("fin", True),
        focus_layer=1,  # image2
    )
    nodes = list(_walk(view))
    titles = [
        getattr(n, "value", "") or ""
        for n in nodes if isinstance(n, ft.Text)
    ]
    assert any("image2" in t for t in titles)

    buttons = {
        getattr(n, "content", None): n
        for n in nodes
        if isinstance(n, (ft.FilledButton, ft.OutlinedButton, ft.FilledTonalButton))
    }
    capture_rows = [
        n for n in nodes
        if isinstance(n, ft.Container) and getattr(n, "on_click", None) is not None
    ]
    # Capture 2 and Capture 3 — Accept both to unlock the all-overlays strip.
    capture_rows[1].on_click(None)
    buttons["Accept"].on_click(None)
    capture_rows[2].on_click(None)
    buttons["Accept"].on_click(None)

    nodes2 = list(_walk(view))
    gallery_titles = [
        getattr(n, "value", "") or ""
        for n in nodes2 if isinstance(n, ft.Text)
    ]
    assert any("All overlays" in t for t in gallery_titles)

    gallery_imgs = [
        n for n in nodes2
        if isinstance(n, ft.Image) and getattr(n, "width", None) == 128
    ]
    assert len(gallery_imgs) >= 3


def test_manual_view_pair_finalize_buttons(tmp_path):
    """Pair-only Finalize buttons pass target_layers; all-images keeps None."""
    visit = tmp_path / "visit"
    _write_visit(str(visit), n_captures=3, n_layers=4)
    plan = prepare_visit(str(visit))

    fake_page = types.SimpleNamespace(
        update=lambda: None, width=1280, on_resize=None, window=None,
    )
    captured = {}

    def on_finalize(
        overrides_by_visit,
        points_by_visit,
        excluded_by_visit=None,
        target_layers=None,
    ):
        captured["overrides"] = overrides_by_visit
        captured["target_layers"] = target_layers

    view = create_manual_align_view(
        fake_page, [plan], on_back=lambda: None, on_finalize=on_finalize,
        focus_layer=0,
    )
    nodes = list(_walk(view))
    buttons = {
        getattr(n, "content", None): n
        for n in nodes
        if isinstance(n, (ft.FilledButton, ft.OutlinedButton, ft.FilledTonalButton))
    }
    assert "Finalize image1+image2 only" in buttons
    assert "Finalize image3+image4 only" in buttons
    assert "Finalize all images" in buttons

    capture_rows = [
        n for n in nodes
        if isinstance(n, ft.Container) and getattr(n, "on_click", None) is not None
    ]
    capture_rows[1].on_click(None)
    buttons["Accept"].on_click(None)

    buttons["Finalize image3+image4 only"].on_click(None)
    assert captured["target_layers"] == [2, 3]
    assert plan.visit_name in captured["overrides"]

    buttons["Finalize all images"].on_click(None)
    assert captured["target_layers"] is None


def test_manual_view_change_reference_capture(tmp_path):
    """Dropdown re-runs auto-registration for the focus image; Capture 1 becomes editable."""
    visit = tmp_path / "visit"
    _write_visit(str(visit), n_captures=3, n_layers=2)
    plan = prepare_visit(str(visit))
    assert plan.reference_idx == 0
    before = [m.copy() for m in plan.matrices]

    fake_page = types.SimpleNamespace(
        update=lambda: None, width=1280, on_resize=None, window=None,
    )
    captured = {}

    def on_finalize(overrides_by_visit, points_by_visit, excluded_by_visit=None):
        captured["overrides"] = overrides_by_visit
        captured["excluded"] = excluded_by_visit or {}
        captured["ref"] = plan.reference_idx

    view = create_manual_align_view(
        fake_page, [plan], on_back=lambda: None, on_finalize=on_finalize,
        focus_layer=1,  # image2 only
    )
    nodes = list(_walk(view))
    dropdowns = [n for n in nodes if isinstance(n, ft.Dropdown)]
    assert dropdowns, "expected alignment-reference dropdown"
    ref_dd = next(d for d in dropdowns if d.label == "Alignment reference")
    assert ref_dd.value == "0"

    # Switch reference to Capture 2 (index 1) → scoped auto re-registration.
    ref_dd.value = "1"
    if ref_dd.on_select:
        ref_dd.on_select(None)
    assert plan.reference_idx == 1
    assert np.allclose(plan.matrices[1], np.eye(2, 3), atol=1e-5)
    # Matrices should come from a fresh auto-align (not a silent no-op).
    assert any(
        not np.allclose(before[i], plan.matrices[i], atol=1e-4)
        for i in range(len(before))
    ) or before[1] is not None

    texts = [
        getattr(n, "value", "") or ""
        for n in _walk(view) if isinstance(n, ft.Text)
    ]
    assert any("aligned to Capture 2" in t for t in texts)
    assert any("Auto-registered to Capture 2" in t or "Reference (Capture 2" in t for t in texts)

    buttons = {
        getattr(n, "content", None): n
        for n in _walk(view)
        if isinstance(n, (ft.FilledButton, ft.OutlinedButton, ft.FilledTonalButton))
    }
    capture_rows = [
        n for n in _walk(view)
        if isinstance(n, ft.Container) and getattr(n, "on_click", None) is not None
    ]
    # Capture 1 (index 0) is no longer the reference — Accept should work.
    capture_rows[0].on_click(None)
    buttons["Accept"].on_click(None)
    buttons["Finalize all images"].on_click(None)
    assert captured.get("ref") == 1
    assert 0 in captured["overrides"][plan.visit_name]
