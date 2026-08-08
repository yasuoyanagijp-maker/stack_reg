import os
import numpy as np
import cv2
import pytest

from app.core.pipeline import (
    prepare_visit,
    finalize_visit,
    run_registration_pipeline,
    discover_visits,
)
from app.core.manual_align import (
    save_session,
    load_session,
    overrides_from_session,
)


def _base_pattern(size=128):
    img = np.full((size, size), 20, np.uint8)
    cv2.line(img, (20, 20), (100, 110), 220, 3)
    cv2.line(img, (110, 20), (30, 100), 200, 2)
    cv2.circle(img, (64, 64), 30, 180, 2)
    cv2.rectangle(img, (40, 40), (88, 88), 160, 1)
    cv2.putText(img, "R", (48, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    return img


def _write_synthetic_visit(root, n_captures=3, n_layers=2, size=128):
    """
    Build a synthetic Visit: each capture is the base pattern shifted/rotated by a
    known small transform, so alignment is a meaningful (recoverable) problem.
    """
    base = _base_pattern(size)
    rng = np.random.default_rng(1)
    transforms = []
    for c in range(n_captures):
        angle = c * 2.0
        tx, ty = c * 3.0, -c * 2.0
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        transforms.append(M)
        cap_dir = os.path.join(root, str(c + 1))
        os.makedirs(cap_dir, exist_ok=True)
        warped = cv2.warpAffine(base, M, (size, size))
        for layer in range(n_layers):
            noisy = np.clip(warped.astype(np.int16) + rng.integers(-6, 6, warped.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(cap_dir, f"layer{layer+1}.jpg"), noisy)
    return transforms


def test_prepare_visit_shapes(tmp_path):
    visit = tmp_path / "visit"
    _write_synthetic_visit(str(visit), n_captures=3, n_layers=2)
    plan = prepare_visit(str(visit))
    assert plan.ref_stack.shape[0] == 3
    assert len(plan.matrices) == 3
    assert len(plan.scores) == 3
    assert plan.scores[0] == 1.0
    assert np.allclose(plan.matrices[0], np.eye(2, 3))
    # low_confidence_indices excludes the anchor
    assert 0 not in plan.low_confidence_indices()
    layer0 = plan.load_layer_stack(0)
    assert layer0.shape[0] == 3
    assert layer0.shape[1:] == plan.ref_stack.shape[1:]


def test_finalize_writes_outputs(tmp_path):
    visit = tmp_path / "P" / "visit"
    _write_synthetic_visit(str(visit), n_captures=3, n_layers=2)
    plan = prepare_visit(str(visit))
    out = tmp_path / "out"
    out.mkdir()
    ok = finalize_visit(plan, patient_name="P", patient_output_dir=str(out))
    assert ok
    files = sorted(os.listdir(out))
    # one averaged image per layer
    assert any(f.endswith("image1.tif") for f in files)
    assert any(f.endswith("image2.tif") for f in files)


def test_matrix_override_changes_output(tmp_path):
    visit = tmp_path / "visit"
    _write_synthetic_visit(str(visit), n_captures=3, n_layers=2)
    plan = prepare_visit(str(visit))

    out_auto = tmp_path / "auto"
    out_auto.mkdir()
    finalize_visit(plan, "P", str(out_auto))

    # Deliberately wrong override for capture index 1 (a large translation).
    bad = plan.matrices[1].copy()
    bad[0, 2] += 200.0
    out_over = tmp_path / "over"
    out_over.mkdir()
    # source_layer=0 means params came from image1; they must still rewrite all layers.
    finalize_visit(
        plan, "P", str(out_over), matrix_overrides={1: bad}, source_layer=0
    )

    auto_files = sorted(f for f in os.listdir(out_auto) if f.endswith(".tif"))
    over_files = sorted(f for f in os.listdir(out_over) if f.endswith(".tif"))
    assert auto_files == over_files
    assert any(f.endswith("image1.tif") for f in over_files)
    assert any(f.endswith("image2.tif") for f in over_files)

    for fname in auto_files:
        a = cv2.imread(os.path.join(out_auto, fname), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(os.path.join(out_over, fname), cv2.IMREAD_GRAYSCALE)
        assert a is not None and b is not None
        assert a.shape == b.shape
        assert not np.array_equal(a, b), f"override should alter {fname}"

    # The plan's original matrices must be untouched by the override.
    assert plan.matrices[1][0, 2] != bad[0, 2]


def test_run_pipeline_end_to_end_with_overrides(tmp_path):
    patient = tmp_path / "Patient"
    _write_synthetic_visit(str(patient / "VisitA"), n_captures=2, n_layers=2)
    out = tmp_path / "out"
    out.mkdir()

    ok = run_registration_pipeline(str(patient), str(out))
    assert ok
    produced = list((out / "Patient").glob("*.tif"))
    assert produced

    # Same run but with an override should still succeed and produce output.
    overrides = {"VisitA": {1: np.eye(2, 3, dtype=np.float32)}}
    out2 = tmp_path / "out2"
    out2.mkdir()
    ok2 = run_registration_pipeline(str(patient), str(out2), overrides_by_visit=overrides)
    assert ok2
    assert list((out2 / "Patient").glob("*.tif"))


def test_session_round_trip(tmp_path):
    session = {
        "patient": "P",
        "confidence_threshold": 0.8,
        "visits": {
            "VisitA": {
                "scores": [1.0, 0.5],
                "auto_matrices": [np.eye(2, 3).tolist(), np.eye(2, 3).tolist()],
                "overrides": {"1": np.array([[1, 0, 5], [0, 1, 3]], np.float32).tolist()},
                "points": {"1": {"ref": [[1, 2], [3, 4], [5, 6]], "src": [[1, 2], [3, 4], [5, 6]]}},
            }
        },
    }
    save_session(str(tmp_path), session)
    loaded = load_session(str(tmp_path))
    assert loaded["patient"] == "P"
    ov = overrides_from_session(loaded)
    assert 1 in ov["VisitA"]
    assert np.allclose(ov["VisitA"][1], np.array([[1, 0, 5], [0, 1, 3]], np.float32))
