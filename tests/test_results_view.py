"""Construction test for the post-registration review screen (headless fake page)."""
import types

import numpy as np
import tifffile
import flet as ft

from app.ui.results_view import create_results_view, _index_output_tifs


def _walk(control):
    yield control
    for attr in ("controls", "content", "options"):
        child = getattr(control, attr, None)
        if isinstance(child, (list, tuple)):
            for c in child:
                if c is not None:
                    yield from _walk(c)
        elif child is not None and hasattr(child, "__dict__"):
            yield from _walk(child)


def test_index_output_tifs(tmp_path):
    out = tmp_path / "Patient"
    out.mkdir()
    rng = np.random.default_rng(0)
    for i in range(1, 5):
        img = rng.integers(0, 255, (32, 32)).astype(np.uint8)
        tifffile.imwrite(str(out / f"P-Avg-Stack_Visit1_image{i}.tif"), img)
    (out / "alignment_session.json").write_text("{}")

    indexed = _index_output_tifs(str(out))
    assert sorted(indexed.keys()) == [1, 2, 3, 4]
    assert "Visit1" in indexed[2]


def test_results_view_dropdown_and_review(tmp_path):
    out = tmp_path / "Patient"
    out.mkdir()
    rng = np.random.default_rng(0)
    for i in range(1, 5):
        img = rng.integers(0, 255, (64, 64)).astype(np.uint8)
        tifffile.imwrite(str(out / f"P-Avg-Stack_Visit1_image{i}.tif"), img)

    fake_page = types.SimpleNamespace(update=lambda: None)
    captured = {}

    view = create_results_view(
        fake_page,
        str(out),
        on_back=lambda: captured.setdefault("back", True),
        on_review_correct=lambda image_num, visit: captured.update(
            {"image_num": image_num, "visit": visit}
        ),
        corrections_summary={"Visit1": [1, 2]},
    )

    nodes = list(_walk(view))
    dropdowns = [n for n in nodes if isinstance(n, ft.Dropdown)]
    assert dropdowns, "expected image dropdown"
    image_dd = next(d for d in dropdowns if d.label == "Result image")
    assert image_dd.value == "1"
    assert {o.key for o in image_dd.options} == {"1", "2", "3", "4"}

    images = [n for n in nodes if isinstance(n, ft.Image)]
    assert len(images) == 1
    assert isinstance(images[0].src, bytes) and images[0].src[:4] == b"\x89PNG"

    texts = [getattr(n, "value", "") or "" for n in nodes if isinstance(n, ft.Text)]
    assert any("Capture 2, 3 manually corrected" in t for t in texts)

    # Switch to image3 via dropdown select handler
    image_dd.value = "3"
    image_dd.on_select(None)
    fname_texts = [
        getattr(n, "value", "") or ""
        for n in _walk(view)
        if isinstance(n, ft.Text) and (getattr(n, "value", "") or "").endswith(".tif")
    ]
    assert any("image3" in t for t in fname_texts)

    review_btn = next(
        n for n in nodes
        if isinstance(n, ft.FilledButton)
        and getattr(n, "content", None) == "Review & Correct"
    )
    review_btn.on_click(None)
    assert captured.get("image_num") == 3
    assert captured.get("visit") == "Visit1"

    # Second entry: change dropdown value without relying on stale state alone.
    # Simulates returning from manual editor then picking another image.
    image_dd.value = "2"
    # Even if on_select were skipped, Review must read the live dropdown.
    review_btn.on_click(None)
    assert captured.get("image_num") == 2

    back_btn = next(n for n in nodes if isinstance(n, ft.IconButton))
    back_btn.on_click(None)
    assert captured.get("back") is True


def test_results_view_preserves_visit_on_image_change(tmp_path):
    """Changing Result image must keep the selected Visit (not jump to first)."""
    out = tmp_path / "Patient"
    out.mkdir()
    rng = np.random.default_rng(2)
    for visit in ("VisitA", "VisitB"):
        for i in range(1, 5):
            img = rng.integers(0, 255, (32, 32)).astype(np.uint8)
            tifffile.imwrite(str(out / f"P-Avg-Stack_{visit}_image{i}.tif"), img)

    fake_page = types.SimpleNamespace(update=lambda: None)
    view = create_results_view(
        fake_page,
        str(out),
        on_back=lambda: None,
        on_review_correct=lambda image_num, visit: None,
    )
    nodes = list(_walk(view))
    image_dd = next(
        n for n in nodes if isinstance(n, ft.Dropdown) and n.label == "Result image"
    )
    visit_dd = next(
        n for n in nodes if isinstance(n, ft.Dropdown) and n.label == "Visit"
    )
    assert visit_dd.visible is True
    assert visit_dd.value == "VisitA"

    visit_dd.value = "VisitB"
    visit_dd.on_select(None)
    assert visit_dd.value == "VisitB"

    image_dd.value = "3"
    image_dd.on_select(None)
    assert visit_dd.value == "VisitB"
    fname_texts = [
        getattr(n, "value", "") or ""
        for n in _walk(view)
        if isinstance(n, ft.Text) and (getattr(n, "value", "") or "").endswith(".tif")
    ]
    assert any("VisitB" in t and "image3" in t for t in fname_texts)


def test_results_view_initial_image_num(tmp_path):
    out = tmp_path / "Patient"
    out.mkdir()
    rng = np.random.default_rng(1)
    for i in range(1, 5):
        img = rng.integers(0, 255, (32, 32)).astype(np.uint8)
        tifffile.imwrite(str(out / f"P-Avg-Stack_Visit1_image{i}.tif"), img)

    fake_page = types.SimpleNamespace(update=lambda: None)
    captured = {}
    view = create_results_view(
        fake_page,
        str(out),
        on_back=lambda: None,
        on_review_correct=lambda image_num, visit: captured.update(
            {"image_num": image_num, "visit": visit}
        ),
        initial_image_num=2,
    )
    nodes = list(_walk(view))
    image_dd = next(
        n for n in nodes if isinstance(n, ft.Dropdown) and n.label == "Result image"
    )
    assert image_dd.value == "2"
    review_btn = next(
        n for n in nodes
        if isinstance(n, ft.FilledButton)
        and getattr(n, "content", None) == "Review & Correct"
    )
    # Change to image4 on the control, then Review without calling on_select.
    image_dd.value = "4"
    review_btn.on_click(None)
    assert captured["image_num"] == 4
