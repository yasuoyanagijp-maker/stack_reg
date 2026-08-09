import numpy as np

from app.core.manual_align import nearest_landmark_index, draw_landmarks
from app.ui.manual_align_view import (
    fit_display_width,
    DISPLAY_W_MAX,
    DISPLAY_W_MIN,
    LEFT_PANEL_W,
)


def test_nearest_landmark_index_hits_closest():
    pts = [[10.0, 10.0], [50.0, 50.0], [80.0, 20.0]]
    assert nearest_landmark_index(pts, 12.0, 11.0, max_dist=10.0) == 0
    assert nearest_landmark_index(pts, 48.0, 52.0, max_dist=10.0) == 1
    assert nearest_landmark_index(pts, 0.0, 0.0, max_dist=5.0) is None


def test_fit_display_width_keeps_three_panels_inside_window():
    # 3 * w + left panel + gutter must not exceed the window width.
    for pw in (900, 1100, 1280, 1600):
        w = fit_display_width(pw)
        assert DISPLAY_W_MIN <= w <= DISPLAY_W_MAX
        assert 3 * w + LEFT_PANEL_W + 80 <= pw + 1


def test_fit_display_width_clamps():
    assert fit_display_width(500) == DISPLAY_W_MIN
    assert fit_display_width(2400) == DISPLAY_W_MAX


def test_draw_landmarks_selected_highlight():
    img = np.zeros((64, 64), np.uint8)
    plain = draw_landmarks(img, [[32, 32]], color=(0, 255, 0), selected_idx=None)
    selected = draw_landmarks(img, [[32, 32]], color=(0, 255, 0), selected_idx=0)
    assert selected.ndim == 3
    # Highlight ring adds cyan (BGR blue channel).
    assert int(selected[..., 0].sum()) > int(plain[..., 0].sum())
