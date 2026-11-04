"""Synthetic landmark data with the same format as ``landmarks.extract``.

ONLY for testing the code path end-to-end (no MediaPipe, no videos). Numbers produced on
synthetic data say nothing about BISINDO and must never be reported as results.

Each synthetic "sign" = idle (hands down) -> raise -> class-specific hand trajectory and
hand shape -> lower -> idle. Durations vary per sample, so natural speed variation exists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _class_params(n_classes: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    params = []
    for c in range(n_classes):
        params.append({
            "amp": rng.uniform(0.01, 0.05, size=(2, 2, 3)),      # hand, axis, harmonic
            "phase": rng.uniform(0, 2 * np.pi, size=(2, 2, 3)),
            "center": rng.uniform([0.0, -0.02], [0.12, 0.15], size=(2, 2)),
            "shape_a": rng.normal(0, 0.03, size=(21, 2)),
            "shape_b": rng.normal(0, 0.03, size=(21, 2)),
            "two_hands": bool(rng.random() < 0.5),
        })
    return params


_BASE_HAND = np.stack([np.cos(np.linspace(0, np.pi, 21)) * 0.04,
                       -np.linspace(0, 0.08, 21)], axis=1)


def make_sample(label: int, signer: int, rng: np.random.Generator, params, fps=30.0) -> dict:
    p = params[label]
    head = int(rng.integers(4, 20))
    tail = int(rng.integers(4, 20))
    ramp = int(rng.integers(5, 9))
    core = int(rng.integers(24, 60))                       # natural speed variation
    N = head + ramp + core + ramp + tail

    s_rng = np.random.default_rng(1000 + signer)            # signer style
    s_scale = s_rng.uniform(0.9, 1.1)
    s_offset = s_rng.normal(0, 0.02, size=2)

    # image-space body: shoulders ~0.2 apart -> 1 shoulder width = 0.2
    sw = 0.2 * s_scale
    sh_c = np.array([0.5, 0.42]) + s_offset
    pose = np.full((N, 33, 4), np.nan, np.float32)
    pose[:, :, 3] = 0.9
    body = {0: [0, -0.9], 11: [-0.5, 0], 12: [0.5, 0], 13: [-0.65, 0.9], 14: [0.65, 0.9],
            23: [-0.35, 1.6], 24: [0.35, 1.6]}

    rest = np.array([[-0.55, 1.9], [0.55, 1.9]])            # wrists at rest (shoulder widths)
    wrists = np.zeros((N, 2, 2))
    shape_mix = np.zeros(N)
    u = np.linspace(0, 1, core)
    active_pos = np.zeros((core, 2, 2))
    for h in range(2):
        for ax in range(2):
            traj = sum(p["amp"][h, ax, k] / 0.2 * np.sin(2 * np.pi * (k + 1) * u + p["phase"][h, ax, k])
                       for k in range(3))
            sign = (-1.0 if (h == 0 and ax == 0) else 1.0)  # left hand mirrored in x
            active_pos[:, h, ax] = sign * p["center"][h, ax] / 0.2 + traj
    if not p["two_hands"]:
        active_pos[:, 0] = rest[0] + np.array([0.1, -0.3])   # non-dominant hand stays low-ish
    start, end = active_pos[0], active_pos[-1]
    for t in range(N):
        if t < head:
            wrists[t] = rest
        elif t < head + ramp:
            w = (t - head + 1) / ramp
            wrists[t] = rest * (1 - w) + start * w
        elif t < head + ramp + core:
            i = t - head - ramp
            wrists[t] = active_pos[i]
            shape_mix[t] = u[i]
        elif t < head + 2 * ramp + core:
            w = (t - head - ramp - core + 1) / ramp
            wrists[t] = end * (1 - w) + rest * w
        else:
            wrists[t] = rest

    def to_img(xy):
        return sh_c + np.asarray(xy) * sw

    for t in range(N):
        for j, xy in body.items():
            pose[t, j, :2] = to_img(xy) + rng.normal(0, 0.002, 2)
        pose[t, 15, :2] = to_img(wrists[t, 0]) + rng.normal(0, 0.003, 2)
        pose[t, 16, :2] = to_img(wrists[t, 1]) + rng.normal(0, 0.003, 2)
        pose[t, :, 2] = 0.0

    hands = np.full((2, N, 21, 3), np.nan, np.float32)
    for h in range(2):
        for t in range(N):
            if wrists[t, h, 1] > 1.5 or rng.random() < 0.03:  # hand low / random dropout -> not detected
                continue
            shape = _BASE_HAND + p["shape_a"] * (1 - shape_mix[t]) + p["shape_b"] * shape_mix[t]
            if h == 0:
                shape = shape * np.array([-1, 1])
            hands[h, t, :, :2] = pose[t, 15 + h, :2] + shape * s_scale + rng.normal(0, 0.002, (21, 2))
            hands[h, t, :, 2] = 0.0
    return {"pose": pose, "lhand": hands[0], "rhand": hands[1], "fps": np.float32(fps),
            "width": np.int32(1000), "height": np.int32(1000)}


def make_dataset(out_dir: str | Path, n_classes=8, n_signers=4, n_samples=6, seed=0) -> pd.DataFrame:
    """Write synthetic npz files and return an index DataFrame like dataset.build_index."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = _class_params(n_classes)
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_signers):
        for c in range(n_classes):
            for k in range(1, n_samples + 1):
                vid = f"signer{s}_label{c}_sample{k}"
                np.savez_compressed(out / f"{vid}.npz", **make_sample(c, s, rng, params))
                rows.append({"video_id": vid, "path": "", "signer": s, "label": c,
                             "gloss": f"SYN{c}", "sample": k, "dup_id": False})
    return pd.DataFrame(rows)
