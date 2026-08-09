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
from app.ui.manual_align_view import create_manual_align_view


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
    assert "Finalize & Save" in buttons
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
    buttons["Finalize & Save"].on_click(None)

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
