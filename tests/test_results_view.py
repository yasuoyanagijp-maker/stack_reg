"""Construction test for the post-finalize results gallery (headless fake page)."""
import types

import numpy as np
import tifffile
import flet as ft

from app.ui.results_view import create_results_view


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


def test_results_view_builds_gallery(tmp_path):
    out = tmp_path / "Patient"
    out.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        img = rng.integers(0, 255, (64, 64)).astype(np.uint8)
        tifffile.imwrite(str(out / f"P-Avg-Stack_Visit1_image{i+1}.tif"), img)
    (out / "alignment_session.json").write_text("{}")  # non-tif must be ignored

    fake_page = types.SimpleNamespace(update=lambda: None)
    captured = {}
    view = create_results_view(
        fake_page, str(out),
        on_back=lambda: captured.setdefault("back", True),
        corrections_summary={"Visit1": [1, 2]},
    )

    nodes = list(_walk(view))
    images = [n for n in nodes if isinstance(n, ft.Image)]
    assert len(images) == 2
    assert all(isinstance(img.src, bytes) and img.src[:4] == b"\x89PNG" for img in images)

    texts = [getattr(n, "value", "") or "" for n in nodes if isinstance(n, ft.Text)]
    assert any("Capture 2, 3 manually corrected" in t for t in texts)
    assert any(t.endswith(".tif") for t in texts)

    back_btn = next(n for n in nodes if isinstance(n, ft.IconButton))
    back_btn.on_click(None)
    assert captured.get("back") is True
