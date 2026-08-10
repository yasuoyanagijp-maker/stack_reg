import numpy as np
import cv2
import pytest

from app.core.registration import (
    estimate_affine_from_correspondences,
    compute_alignment_cc,
    seed_correspondences_from_matrix,
    extract_feature_correspondences,
    correspondence_residuals,
    filter_correspondences_by_residual,
    nudge_affine_matrix,
    invert_affine_2x3,
    compose_affine_2x3,
    rebase_affine_matrices,
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


def test_seed_correspondences_round_trip():
    theta = np.deg2rad(5)
    M = np.array([[np.cos(theta), -np.sin(theta), 8.0],
                  [np.sin(theta), np.cos(theta), -4.0]], np.float32)
    ref_pts, src_pts = seed_correspondences_from_matrix(M, height=200, width=200)
    assert len(ref_pts) == 6 and len(src_pts) == 6
    est = estimate_affine_from_correspondences(np.array(ref_pts), np.array(src_pts))
    assert np.abs(est - M).max() < 1e-3


def test_seed_correspondences_n_points_clamped():
    M = np.eye(2, 3, dtype=np.float32)
    ref3, src3 = seed_correspondences_from_matrix(M, 100, 100, n_points=3)
    ref8, src8 = seed_correspondences_from_matrix(M, 100, 100, n_points=8)
    assert len(ref3) == 3 and len(src3) == 3
    assert len(ref8) == 8 and len(src8) == 8


def test_nudge_affine_translation():
    M = np.eye(2, 3, dtype=np.float32)
    nudged = nudge_affine_matrix(M, dx=10.0, dy=-5.0)
    # Overlay content moves +dx,+dy → WARP_INVERSE translation becomes -dx,-dy.
    assert abs(nudged[0, 2] - (-10.0)) < 1e-5
    assert abs(nudged[1, 2] - 5.0) < 1e-5


def test_filter_correspondences_by_residual():
    M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], np.float32)
    ref = np.array([[10, 10], [50, 20], [80, 90], [30, 70]], np.float32)
    src = (M @ np.hstack([ref, np.ones((4, 1))]).T).T.astype(np.float32)
    src[3] += np.array([40.0, 40.0], np.float32)  # outlier
    resid = correspondence_residuals(ref, src, M)
    assert resid[3] > 20
    ref_in, src_in, keep = filter_correspondences_by_residual(ref, src, M, max_residual=5.0)
    assert keep.sum() == 3
    assert ref_in.shape[0] == 3


def test_extract_feature_correspondences_on_shifted_image():
    ref = _structured_image(256)
    M = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -8.0]], np.float32)
    src = _make_source(ref, M)
    extracted = extract_feature_correspondences(ref, src)
    assert extracted is not None
    ref_pts, src_pts = extracted
    assert ref_pts.shape[0] >= 3
    resid = correspondence_residuals(ref_pts, src_pts, M)
    # Most true matches should fit the known transform tightly.
    assert np.median(resid) < 3.0


def test_rebase_affine_matrices_new_reference_is_identity():
    mats = [
        np.eye(2, 3, dtype=np.float32),
        np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -2.0]], np.float32),
        np.array([[1.0, 0.0, -3.0], [0.0, 1.0, 4.0]], np.float32),
    ]
    rebased = rebase_affine_matrices(mats, 2)
    assert np.allclose(rebased[2], np.eye(2, 3), atol=1e-5)
    # Round-trip: rebasing back to 0 restores original (within float error).
    restored = rebase_affine_matrices(rebased, 0)
    for a, b in zip(mats, restored):
        assert np.allclose(a, b, atol=1e-4)


def test_rebase_preserves_point_mapping():
    """M'_i @ p_new_ref == M_i @ p_old_ref when p_new = M_new @ p_old."""
    mats = [
        np.eye(2, 3, dtype=np.float32),
        np.array([[0.98, -0.05, 4.0], [0.05, 0.98, -3.0]], np.float32),
        np.array([[1.0, 0.02, -6.0], [-0.02, 1.0, 5.0]], np.float32),
    ]
    new_ref = 1
    rebased = rebase_affine_matrices(mats, new_ref)
    p_old = np.array([40.0, 55.0, 1.0], np.float64)
    M_new = np.vstack([mats[new_ref], [0, 0, 1]])
    p_new = M_new @ p_old
    for i in range(len(mats)):
        Mi = np.vstack([mats[i], [0, 0, 1]])
        Mi2 = np.vstack([rebased[i], [0, 0, 1]])
        assert np.allclose(Mi @ p_old, Mi2 @ p_new, atol=1e-4)


def test_compose_invert_roundtrip():
    M = np.array([[0.97, -0.1, 8.0], [0.1, 0.97, -4.0]], np.float32)
    assert np.allclose(
        compose_affine_2x3(M, invert_affine_2x3(M)),
        np.eye(2, 3),
        atol=1e-5,
    )


def test_calculate_affine_with_non_zero_reference():
    from app.core.registration import calculate_affine_transformations

    ref = _structured_image(128)
    stack = np.stack([
        _make_source(ref, np.array([[1, 0, 5], [0, 1, -3]], np.float32)),
        ref,
        _make_source(ref, np.array([[1, 0, -4], [0, 1, 6]], np.float32)),
    ], axis=0)
    mats, scores = calculate_affine_transformations(
        stack, return_scores=True, reference_idx=1
    )
    assert np.allclose(mats[1], np.eye(2, 3), atol=1e-5)
    assert scores[1] == 1.0
    assert len(mats) == 3
    # Other captures should have a usable confidence after alignment.
    assert scores[0] > 0.5 and scores[2] > 0.5

