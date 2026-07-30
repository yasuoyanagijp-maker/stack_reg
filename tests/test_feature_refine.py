import numpy as np
import cv2

from app.core.registration import (
    refine_affine_feature_based,
    compute_alignment_cc,
)
from app.core.pipeline import auto_refine_matrices


def _feature_rich_image(size=256, seed=0):
    """A textured image with many corners so ORB has plenty of keypoints."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 15, np.uint8)
    for _ in range(60):
        x, y = rng.integers(10, size - 10, 2)
        r = int(rng.integers(3, 12))
        val = int(rng.integers(120, 255))
        cv2.rectangle(img, (x - r, y - r), (x + r, y + r), val, -1)
    for _ in range(30):
        p1 = tuple(rng.integers(0, size, 2).tolist())
        p2 = tuple(rng.integers(0, size, 2).tolist())
        cv2.line(img, p1, p2, int(rng.integers(80, 200)), 2)
    return img


def _source_from(ref, M):
    M3 = np.vstack([M, [0, 0, 1]])
    Minv = np.linalg.inv(M3)[:2].astype(np.float32)
    h, w = ref.shape[:2]
    return cv2.warpAffine(ref, Minv, (w, h),
                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


def test_feature_refine_recovers_transform():
    ref = _feature_rich_image()
    theta = np.deg2rad(5)
    M = np.array([[np.cos(theta), -np.sin(theta), 18.0],
                  [np.sin(theta), np.cos(theta), -12.0]], np.float32)
    src = _source_from(ref, M)

    result = refine_affine_feature_based(ref, src)
    assert result is not None
    matrix, cc = result
    assert matrix.shape == (2, 3)
    assert cc > 0.8
    # translation recovered within a few pixels
    assert abs(matrix[0, 2] - M[0, 2]) < 4
    assert abs(matrix[1, 2] - M[1, 2]) < 4


def test_feature_refine_returns_none_without_features():
    blank = np.zeros((128, 128), np.uint8)
    assert refine_affine_feature_based(blank, blank) is None


def test_auto_refine_adopts_only_when_better():
    ref_stack = np.stack([_feature_rich_image(seed=1)] * 3, axis=0)
    good = np.array([[1, 0, 5], [0, 1, 3]], np.float32)
    matrices = [np.eye(2, 3, dtype=np.float32), np.eye(2, 3, dtype=np.float32), np.eye(2, 3, dtype=np.float32)]
    scores = [1.0, 0.5, 0.9]  # capture 1 low, capture 2 ok

    calls = {"n": 0}

    def fake_refine(_ref, _src):
        calls["n"] += 1
        return good, 0.95  # a big improvement

    logs = []
    refined = auto_refine_matrices(ref_stack, matrices, scores,
                                   refine_fn=fake_refine, log=logs.append)
    # Only the below-threshold capture (index 1) is refined.
    assert refined == [1]
    assert calls["n"] == 1
    assert np.allclose(matrices[1], good)
    assert scores[1] == 0.95
    # capture 2 untouched
    assert np.allclose(matrices[2], np.eye(2, 3))


def test_auto_refine_rejects_worse_candidate():
    ref_stack = np.stack([_feature_rich_image(seed=2)] * 2, axis=0)
    matrices = [np.eye(2, 3, dtype=np.float32), np.eye(2, 3, dtype=np.float32)]
    scores = [1.0, 0.5]

    def fake_refine(_ref, _src):
        return np.array([[1, 0, 99], [0, 1, 99]], np.float32), 0.51  # barely better, < min_improvement

    refined = auto_refine_matrices(ref_stack, matrices, scores, refine_fn=fake_refine)
    assert refined == []
    assert np.allclose(matrices[1], np.eye(2, 3))
    assert scores[1] == 0.5


def test_auto_refine_handles_none():
    ref_stack = np.stack([_feature_rich_image(seed=3)] * 2, axis=0)
    matrices = [np.eye(2, 3, dtype=np.float32), np.eye(2, 3, dtype=np.float32)]
    scores = [1.0, 0.4]
    refined = auto_refine_matrices(ref_stack, matrices, scores, refine_fn=lambda a, b: None)
    assert refined == []
    assert scores[1] == 0.4
