import numpy as np
import cv2
import pytest

from app.core.registration import (
    estimate_affine_from_correspondences,
    compute_alignment_cc,
)


def _structured_image(size=200):
    img = np.zeros((size, size), np.uint8)
    cv2.rectangle(img, (40, 40), (120, 90), 200, -1)
    cv2.circle(img, (150, 150), 25, 255, -1)
    cv2.putText(img, "A", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, 150, 3)
    return img


def _make_source(ref, M):
    """Create a source whose WARP_INVERSE_MAP by M reproduces ref."""
    M3 = np.vstack([M, [0, 0, 1]])
    Minv = np.linalg.inv(M3)[:2].astype(np.float32)
    h, w = ref.shape[:2]
    return cv2.warpAffine(ref, Minv, (w, h),
                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


def test_estimate_three_points_exact():
    theta = np.deg2rad(6)
    M = np.array([[np.cos(theta), -np.sin(theta), 10.0],
                  [np.sin(theta), np.cos(theta), -7.0]], np.float32)
    ref = np.array([[40, 40], [120, 90], [150, 150]], np.float32)
    src = (M @ np.hstack([ref, np.ones((3, 1))]).T).T
    est = estimate_affine_from_correspondences(ref, src)
    assert np.abs(est - M).max() < 1e-3


def test_estimate_four_points_least_squares():
    theta = np.deg2rad(6)
    M = np.array([[np.cos(theta), -np.sin(theta), 10.0],
                  [np.sin(theta), np.cos(theta), -7.0]], np.float32)
    ref = np.array([[40, 40], [120, 90], [150, 150], [60, 160]], np.float32)
    src = (M @ np.hstack([ref, np.ones((4, 1))]).T).T
    est = estimate_affine_from_correspondences(ref, src)
    assert np.abs(est - M).max() < 1e-2


def test_estimate_requires_three_points():
    with pytest.raises(ValueError):
        estimate_affine_from_correspondences(
            np.array([[0, 0], [1, 1]]), np.array([[0, 0], [1, 1]])
        )


def test_estimate_mismatched_counts():
    with pytest.raises(ValueError):
        estimate_affine_from_correspondences(
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([[0, 0], [1, 1]]),
        )


def test_cc_prefers_correct_matrix():
    ref = _structured_image()
    theta = np.deg2rad(8)
    M = np.array([[np.cos(theta), -np.sin(theta), 12.0],
                  [np.sin(theta), np.cos(theta), -9.0]], np.float32)
    src = _make_source(ref, M)
    identity = np.eye(2, 3, dtype=np.float32)
    assert compute_alignment_cc(ref, src, M) > 0.9
    assert compute_alignment_cc(ref, src, M) > compute_alignment_cc(ref, src, identity)
