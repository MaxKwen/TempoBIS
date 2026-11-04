"""Turn raw MediaPipe landmarks into a per-frame feature matrix.

Per frame we keep (x, y) of
  * 7 upper-body pose points: nose, shoulders, elbows, wrists
  * 21 left-hand and 21 right-hand points
plus 2 flags saying whether each hand was actually detected (not filled in).
That gives D = (7 + 42) * 2 + 2 = 100 features.

Normalisation (per video, so motion is preserved):
  * x is multiplied by width/height so both axes share one unit
  * origin = median shoulder midpoint, unit = median shoulder width
Missing hands: short interior gaps are linearly interpolated; longer gaps / frames
outside the detected range collapse the hand onto the pose wrist (hand is "down" or
out of view), so the hand trajectory stays physically consistent with the arm.
"""
from __future__ import annotations

import numpy as np

POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
N_POSE, N_HAND = len(POSE_IDX), 21
N_POINTS = N_POSE + 2 * N_HAND
FEAT_DIM = N_POINTS * 2 + 2
# index of points inside the 49-point layout
P_LSHOULDER, P_RSHOULDER, P_LWRIST, P_RWRIST = 1, 2, 5, 6
LHAND_SLICE = slice(N_POSE, N_POSE + N_HAND)
RHAND_SLICE = slice(N_POSE + N_HAND, N_POINTS)


def _interp_time(arr: np.ndarray, max_gap: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaN rows of (T, ...) over time.

    Returns (filled, ok) where ok marks rows that are observed or inside a gap of at
    most ``max_gap`` frames. Rows not ok are left NaN.
    """
    T = arr.shape[0]
    flat = arr.reshape(T, -1).copy()
    obs = ~np.isnan(flat[:, 0])
    ok = obs.copy()
    if obs.sum() == 0:
        return flat.reshape(arr.shape), ok
    idx = np.flatnonzero(obs)
    t = np.arange(T)
    if max_gap is None:
        ok[:] = True
        for c in range(flat.shape[1]):
            flat[:, c] = np.interp(t, idx, flat[idx, c])
        return flat.reshape(arr.shape), ok
    # interior gaps only, bounded length
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a) / (b - a)
            flat[a + 1:b] = flat[a] * (1 - w)[:, None] + flat[b] * w[:, None]
            ok[a + 1:b] = True
    return flat.reshape(arr.shape), ok


def frame_features(raw: dict, hand_fill: str = "interpolate", max_gap: int = 6) -> np.ndarray:
    """raw landmarks dict -> (T, FEAT_DIM) float32 matrix."""
    pose = raw["pose"][:, POSE_IDX, :2].astype(np.float64)          # (T, 7, 2)
    lh = raw["lhand"][:, :, :2].astype(np.float64)                    # (T, 21, 2)
    rh = raw["rhand"][:, :, :2].astype(np.float64)
    T = pose.shape[0]
    w, h = float(raw.get("width", 0) or 0), float(raw.get("height", 0) or 0)
    aspect = w / h if w > 0 and h > 0 else 1.0
    for a in (pose, lh, rh):
        a[..., 0] *= aspect

    l_det = ~np.isnan(lh[:, 0, 0])
    r_det = ~np.isnan(rh[:, 0, 0])

    pose, pose_ok = _interp_time(pose)                               # full interpolation + edge hold
    if not pose_ok.any():
        pose = np.zeros_like(pose)

    hands = []
    for hand, wrist_p in ((lh, P_LWRIST), (rh, P_RWRIST)):
        if hand_fill == "zero":
            filled, ok = hand.copy(), ~np.isnan(hand[:, 0, 0])
        else:
            filled, ok = _interp_time(hand, max_gap=max_gap)
        # not recoverable -> collapse onto the pose wrist (or 0 if pose missing too)
        wrist = pose[:, wrist_p, :]
        filled[~ok] = wrist[~ok][:, None, :] if hand_fill != "zero" else 0.0
        filled = np.nan_to_num(filled)
        hands.append(filled)
    lh, rh = hands

    pts = np.concatenate([pose, lh, rh], axis=1)                     # (T, 49, 2)
    # per-video normalisation
    ls, rs = pts[:, P_LSHOULDER], pts[:, P_RSHOULDER]
    if pose_ok.any():
        center = np.median((ls + rs) / 2, axis=0)
        scale = float(np.median(np.linalg.norm(ls - rs, axis=1)))
    else:
        center = np.array([0.5 * aspect, 0.5])
        scale = 0.0
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 0.3  # rough fallback when shoulders are not visible
    pts = (pts - center) / scale

    flags = np.stack([l_det, r_det], axis=1).astype(np.float64)
    feats = np.concatenate([pts.reshape(T, -1), flags], axis=1)
    return feats.astype(np.float32)


def hand_points(feats: np.ndarray) -> np.ndarray:
    """(T, 2, 22, 2): per hand, its 21 points plus the pose wrist."""
    pts = feats[:, :N_POINTS * 2].reshape(-1, N_POINTS, 2)
    left = np.concatenate([pts[:, LHAND_SLICE], pts[:, [P_LWRIST]]], axis=1)
    right = np.concatenate([pts[:, RHAND_SLICE], pts[:, [P_RWRIST]]], axis=1)
    return np.stack([left, right], axis=1)


def wrist_heights(feats: np.ndarray) -> np.ndarray:
    """(T, 2) vertical position of left/right pose wrist, in shoulder widths below the shoulders."""
    pts = feats[:, :N_POINTS * 2].reshape(-1, N_POINTS, 2)
    return pts[:, [P_LWRIST, P_RWRIST], 1]
