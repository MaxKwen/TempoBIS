"""Temporal normalisation (baseline + proposed) and controlled speed perturbations.

All functions work on a per-frame feature matrix ``seq`` of shape (N, D) and use linear
interpolation at fractional frame positions, so they never need the video again.

Normalisation methods (all output exactly ``T`` timesteps):
  uniform       - T evenly spaced positions over the whole clip (baseline)
  trim_uniform  - drop low-motion head/tail (by cumulative motion energy), then uniform
                  (ablation: trimming only)
  motion_aware  - positions are spaced by a blend of motion density and uniform density:
                  segments with more hand motion receive more timesteps (proposed)

Perturbations (test-time stress test):
  global_speed  - the whole clip is played s x faster (s > 1 => fewer frames)
  core_speed    - only the active signing segment (hands raised) changes speed; the idle
                  head/tail keep their length. The active segment is found from wrist
                  height, NOT from motion energy, so the perturbation is not defined by
                  the same signal the proposed method uses.
  warp_speed    - uneven tempo inside the signing segment: first half s x, second half 1/s x
  hold_pause    - the sign is frozen at its peak (highest wrist) so the signing segment lasts
                  1/s times as long; only s < 1 (a pause cannot speed a sign up)
"""
from __future__ import annotations

import numpy as np

from .features import hand_points, wrist_heights


# ------------------------------------------------------------------ primitives
def interp_at(seq: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Sample (N, D) at fractional frame positions ``times`` (clipped to [0, N-1])."""
    N = seq.shape[0]
    if N == 1:
        return np.repeat(seq, len(times), axis=0)
    t = np.clip(np.asarray(times, dtype=np.float64), 0, N - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, N - 1)
    w = (t - i0)[:, None]
    return (seq[i0] * (1 - w) + seq[i1] * w).astype(seq.dtype)


def motion_energy(seq: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Hand speed per inter-frame segment, shape (N-1,).

    Speed = mean displacement of the 22 hand points (21 landmarks + wrist), summed over
    both hands, smoothed with a centred moving average of ``smooth`` segments, minus the
    10th-percentile value (jitter floor), clipped at 0.
    """
    if seq.shape[0] < 2:
        return np.zeros(0)
    hp = hand_points(seq).astype(np.float64)                 # (N, 2, 22, 2)
    disp = np.linalg.norm(np.diff(hp, axis=0), axis=-1)      # (N-1, 2, 22)
    e = disp.mean(axis=2).sum(axis=1)
    if smooth and smooth > 1 and len(e) > 1:
        k = min(smooth, len(e))
        pad = k // 2
        ep = np.pad(e, (pad, k - 1 - pad), mode="edge")
        e = np.convolve(ep, np.ones(k) / k, mode="valid")
    # remove the landmark-jitter floor so idle frames carry (almost) no motion mass
    e = np.clip(e - np.quantile(e, 0.1), 0.0, None)
    return e


def _positions_from_density(seg_density: np.ndarray, T: int) -> np.ndarray:
    """Invert the cumulative density over frame positions 0..N-1 at T evenly spaced levels."""
    N = len(seg_density) + 1
    c = np.concatenate([[0.0], np.cumsum(seg_density)])
    c /= c[-1]
    levels = np.linspace(0.0, 1.0, T)
    # c is non-decreasing; strictly increasing when every segment has positive density
    return np.interp(levels, c, np.arange(N, dtype=np.float64))


# ------------------------------------------------------------------ normalisation
def uniform(seq: np.ndarray, T: int) -> np.ndarray:
    return interp_at(seq, np.linspace(0, seq.shape[0] - 1, T))


def motion_aware(seq: np.ndarray, T: int, alpha: float = 0.3, smooth: int = 5) -> np.ndarray:
    N = seq.shape[0]
    if N < 3:
        return uniform(seq, T)
    e = motion_energy(seq, smooth)
    total = e.sum()
    if not np.isfinite(total) or total <= 1e-9:
        return uniform(seq, T)
    density = (1 - alpha) * e / total + alpha / (N - 1)
    return interp_at(seq, _positions_from_density(density, T))


def trim_bounds(seq: np.ndarray, quantiles=(0.02, 0.98), smooth: int = 5) -> tuple[float, float]:
    """Fractional frame positions where cumulative motion energy crosses the quantiles."""
    N = seq.shape[0]
    e = motion_energy(seq, smooth)
    if N < 3 or e.sum() <= 1e-9:
        return 0.0, float(N - 1)
    c = np.concatenate([[0.0], np.cumsum(e)]) / e.sum()
    lo, hi = np.interp(quantiles, c, np.arange(N, dtype=np.float64))
    if hi - lo < 1.0:
        return 0.0, float(N - 1)
    return float(lo), float(hi)


def trim_uniform(seq: np.ndarray, T: int, quantiles=(0.02, 0.98), smooth: int = 5) -> np.ndarray:
    lo, hi = trim_bounds(seq, quantiles, smooth)
    return interp_at(seq, np.linspace(lo, hi, T))


def normalize(seq: np.ndarray, method: str, cfg_t: dict) -> np.ndarray:
    T = int(cfg_t["length"])
    if method == "uniform":
        return uniform(seq, T)
    if method == "trim_uniform":
        return trim_uniform(seq, T, tuple(cfg_t["trim_quantiles"]), cfg_t["motion_smooth"])
    if method == "motion_aware":
        return motion_aware(seq, T, cfg_t["motion_alpha"], cfg_t["motion_smooth"])
    raise ValueError(f"unknown temporal method: {method}")


# ------------------------------------------------------------------ perturbations
def global_speed(seq: np.ndarray, factor: float) -> np.ndarray:
    """Play the whole clip ``factor`` x faster at the same frame rate."""
    if factor == 1.0:
        return seq
    N = seq.shape[0]
    n_new = max(2, int(round((N - 1) / factor)) + 1)
    return interp_at(seq, np.linspace(0, N - 1, n_new))


def active_segment(seq: np.ndarray, min_rise: float = 0.3, rel_rise: float = 0.25) -> tuple[int, int]:
    """First/last frame where a wrist is clearly raised above its own rest height.

    Per wrist: rest = 90th percentile of its height (y grows downward), range = rest - 5th
    percentile. A frame is active when the wrist is higher than rest by at least
    max(min_rise, rel_rise * range) shoulder widths. Uses only wrist height, never motion.
    """
    N = seq.shape[0]
    y = wrist_heights(seq).astype(np.float64)                 # (N, 2)
    rest = np.percentile(y, 90, axis=0)
    top = np.percentile(y, 5, axis=0)
    margin = np.maximum(min_rise, rel_rise * (rest - top))
    raised = (y < rest - margin)
    # ignore a wrist that never really moves (e.g. a one-handed sign)
    raised[:, (rest - top) < min_rise] = False
    active = np.flatnonzero(raised.any(axis=1))
    if len(active) < 2 or active[-1] - active[0] < 2:
        return 0, N - 1
    return int(active[0]), int(active[-1])


def core_speed(seq: np.ndarray, factor: float, segment: tuple[int, int] | None = None) -> np.ndarray:
    """Change the speed of the active segment only; idle head/tail are untouched."""
    if factor == 1.0:
        return seq
    a, b = segment if segment is not None else active_segment(seq)
    core = seq[a:b + 1]
    n_new = max(2, int(round((b - a) / factor)) + 1)
    core_new = interp_at(core, np.linspace(0, len(core) - 1, n_new))
    return np.concatenate([seq[:a], core_new, seq[b + 1:]], axis=0)


def _resample_len(part: np.ndarray, factor: float) -> np.ndarray:
    """Play ``part`` ``factor`` x faster (same frame rate)."""
    n_new = max(2, int(round((len(part) - 1) / factor)) + 1)
    return interp_at(part, np.linspace(0, len(part) - 1, n_new))


def warp_speed(seq: np.ndarray, factor: float, segment: tuple[int, int] | None = None) -> np.ndarray:
    """Uneven tempo inside the signing segment: first half ``factor`` x, second half ``1/factor`` x.

    Idle head/tail are untouched. factor < 1 = slow start, rushed end; factor > 1 = the reverse.
    """
    if factor == 1.0:
        return seq
    a, b = segment if segment is not None else active_segment(seq)
    m = (a + b) // 2
    first = _resample_len(seq[a:m + 1], factor)
    second = _resample_len(seq[m:b + 1], 1.0 / factor)
    return np.concatenate([seq[:a], first, second[1:], seq[b + 1:]], axis=0)


def hold_frame(seq: np.ndarray, segment: tuple[int, int] | None = None) -> int:
    """Frame inside the signing segment where a wrist is highest (the natural place to hold)."""
    a, b = segment if segment is not None else active_segment(seq)
    y = wrist_heights(seq[a:b + 1]).astype(np.float64)       # y grows downward
    return a + int(np.argmin(y.min(axis=1)))


def hold_pause(seq: np.ndarray, factor: float, segment: tuple[int, int] | None = None) -> np.ndarray:
    """Freeze the sign at its peak so the signing segment lasts ``1/factor`` times as long.

    Only slows down (factor < 1): a pause cannot make a sign faster. factor >= 1 returns ``seq``.
    """
    if factor >= 1.0:
        return seq
    a, b = segment if segment is not None else active_segment(seq)
    n_hold = int(round((b - a + 1) * (1.0 / factor - 1.0)))
    if n_hold <= 0:
        return seq
    h = hold_frame(seq, (a, b))
    return np.concatenate([seq[:h + 1], np.repeat(seq[h:h + 1], n_hold, axis=0), seq[h + 1:]], axis=0)


PERTURBATIONS = ("global", "core", "warp", "hold")


def applicable(kind: str, factor: float) -> bool:
    """Whether a (perturbation, factor) cell is evaluated. Holds only slow a sign down."""
    return not (kind == "hold" and factor > 1.0)


def perturb(seq: np.ndarray, kind: str, factor: float) -> np.ndarray:
    if kind == "global":
        return global_speed(seq, factor)
    if kind == "core":
        return core_speed(seq, factor)
    if kind == "warp":
        return warp_speed(seq, factor)
    if kind == "hold":
        return hold_pause(seq, factor)
    raise ValueError(f"unknown perturbation: {kind}")


def random_speed(seq: np.ndarray, rng: np.random.Generator, low=0.6, high=1.6) -> np.ndarray:
    """Training-time augmentation for the 'uniform + speed aug' comparator."""
    f = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    kind = "global" if rng.random() < 0.5 else "core"
    return perturb(seq, kind, f)


def random_tempo(seq: np.ndarray, rng: np.random.Generator, low=0.25, high=4.0) -> np.ndarray:
    """Training-time augmentation for the 'uniform + full aug' comparator.

    Covers the stress test's segment-level perturbations (core, warp, hold) over its whole
    range. 'global' is left out: uniform resampling is invariant to it, so it adds nothing.
    """
    f = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    kind = ("core", "warp", "hold")[int(rng.integers(3))]
    if kind == "hold" and f > 1.0:
        f = 1.0 / f                                         # a hold can only slow a sign down
    return perturb(seq, kind, f)
