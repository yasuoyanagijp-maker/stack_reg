"""
Helpers that support the manual corresponding-point correction workflow:

- rendering reference / source captures (and an alignment overlay) for display,
- encoding images for Flet's ``src_base64``,
- persisting the per-visit alignment (auto matrices, scores and manual overrides)
  so a corrected run is reproducible and can be re-driven headlessly.
"""
import base64
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np


def to_base64_png(img: np.ndarray, max_side: Optional[int] = None) -> str:
    """
    Encode a grayscale or BGR image as a base64 PNG string suitable for
    ``ft.Image(src_base64=...)``. Optionally downscale so the longest side is at
    most ``max_side`` px (keeps the UI responsive for 4x-enlarged captures).
    """
    out = img
    if max_side is not None:
        h, w = out.shape[:2]
        longest = max(h, w)
        if longest > max_side:
            scale = max_side / float(longest)
            out = cv2.resize(out, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise ValueError("Failed to PNG-encode image for display.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def make_overlay(reference: np.ndarray, source: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Build a two-colour overlay to judge alignment quality: the reference capture
    is drawn in green and the ``matrix``-warped source in magenta. Where the two
    agree the result is near-white/grey; coloured fringes reveal misalignment.

    ``matrix`` maps reference coords -> source coords (``WARP_INVERSE_MAP``).
    """
    h, w = reference.shape[:2]
    warped = cv2.warpAffine(
        source.astype(np.float32), matrix.astype(np.float32), (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped = np.clip(warped, 0, 255).astype(np.uint8)
    ref = reference.astype(np.uint8)

    # OpenCV is BGR: magenta = blue + red channels, green = green channel.
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[..., 0] = warped   # Blue
    overlay[..., 1] = ref      # Green
    overlay[..., 2] = warped   # Red
    return overlay


def draw_landmarks(img: np.ndarray, points: List[List[float]], color=(0, 255, 255)) -> np.ndarray:
    """Return a BGR copy of ``img`` with numbered landmark markers drawn on it."""
    if img.ndim == 2:
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = img.copy()
    r = max(4, int(round(max(img.shape[:2]) * 0.006)))
    for i, (x, y) in enumerate(points):
        c = (int(round(x)), int(round(y)))
        cv2.circle(canvas, c, r, color, 2)
        cv2.circle(canvas, c, 1, color, -1)
        cv2.putText(canvas, str(i + 1), (c[0] + r + 2, c[1] - r),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.4, r * 0.12), color, 2, cv2.LINE_AA)
    return canvas


SESSION_FILENAME = "alignment_session.json"


def session_path(patient_output_dir: str) -> str:
    return os.path.join(patient_output_dir, SESSION_FILENAME)


def save_session(patient_output_dir: str, session: Dict) -> str:
    """
    Persist an alignment session as JSON. ``session`` is expected to hold, per
    visit, the auto matrices, confidence scores, manual overrides and the picked
    corresponding points. NumPy arrays are converted to plain lists.
    """
    os.makedirs(patient_output_dir, exist_ok=True)
    path = session_path(patient_output_dir)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(session), f, indent=2)
    return path


def load_session(patient_output_dir: str) -> Optional[Dict]:
    """Load a previously saved alignment session, or ``None`` if absent."""
    path = session_path(patient_output_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def overrides_from_session(session: Dict) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Reconstruct the ``overrides_by_visit`` structure consumed by
    ``run_registration_pipeline`` from a saved session.
    """
    result: Dict[str, Dict[int, np.ndarray]] = {}
    for visit_name, data in session.get("visits", {}).items():
        overrides = data.get("overrides", {})
        if not overrides:
            continue
        result[visit_name] = {
            int(idx): np.asarray(mat, dtype=np.float32)
            for idx, mat in overrides.items()
        }
    return result
