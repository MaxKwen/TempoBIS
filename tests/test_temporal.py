import numpy as np
import pytest

from tempobis import features as F
from tempobis import temporal as tm
from tempobis.landmarks import assign_hands
from tempobis.synthetic import _class_params, make_sample

CFG_T = dict(length=32, motion_alpha=0.3, motion_smooth=5, trim_quantiles=[0.02, 0.98])


@pytest.fixture(scope="module")
def seq():
    rng = np.random.default_rng(0)
    return F.frame_features(make_sample(2, 1, rng, _class_params(8)))


def test_features_shape_and_finite(seq):
    assert seq.shape[1] == F.FEAT_DIM == 100
    assert np.isfinite(seq).all()


def test_features_normalised_to_shoulders(seq):
    pts = seq[:, :F.N_POINTS * 2].reshape(-1, F.N_POINTS, 2)
    width = np.median(np.linalg.norm(pts[:, F.P_LSHOULDER] - pts[:, F.P_RSHOULDER], axis=1))
    assert width == pytest.approx(1.0, abs=0.05)


def test_interp_at_endpoints(seq):
    out = tm.interp_at(seq, np.array([0.0, len(seq) - 1.0]))
    np.testing.assert_allclose(out[0], seq[0])
    np.testing.assert_allclose(out[1], seq[-1])


@pytest.mark.parametrize("method", ["uniform", "trim_uniform", "motion_aware"])
def test_methods_output_fixed_length(seq, method):
    assert tm.normalize(seq, method, CFG_T).shape == (32, seq.shape[1])
    assert tm.normalize(seq[:2], method, CFG_T).shape == (32, seq.shape[1])


def test_motion_aware_alpha_one_is_uniform(seq):
    np.testing.assert_allclose(tm.motion_aware(seq, 32, alpha=1.0), tm.uniform(seq, 32), atol=1e-5)


def test_motion_aware_samples_more_in_motion(seq):
    e = tm.motion_energy(seq)
    density = 0.7 * e / e.sum() + 0.3 / len(e)
    pos = tm._positions_from_density(density, 32)
    assert np.all(np.diff(pos) >= -1e-9)
    a, b = tm.active_segment(seq)
    frac_in_active = np.mean((pos >= a) & (pos <= b))
    assert frac_in_active > (b - a) / (len(seq) - 1)


def test_global_speed_lengths(seq):
    n = len(seq)
    assert len(tm.global_speed(seq, 0.5)) == pytest.approx(2 * (n - 1) + 1, abs=1)
    assert len(tm.global_speed(seq, 2.0)) == pytest.approx((n - 1) / 2 + 1, abs=1)
    assert tm.global_speed(seq, 1.0) is seq


def test_core_speed_keeps_idle(seq):
    a, b = tm.active_segment(seq)
    fast = tm.core_speed(seq, 1.5, (a, b))
    np.testing.assert_allclose(fast[:a], seq[:a])
    np.testing.assert_allclose(fast[-(len(seq) - b - 1):], seq[b + 1:])
    assert len(fast) < len(seq)


def test_uniform_absorbs_global_slowdown(seq):
    # slowing the whole clip by 2x then resampling uniformly lands on the same positions
    ref = tm.uniform(seq, 32)
    np.testing.assert_allclose(tm.uniform(tm.global_speed(seq, 0.5), 32), ref, atol=1e-4)


def test_active_segment_within_bounds(seq):
    a, b = tm.active_segment(seq)
    assert 0 <= a < b <= len(seq) - 1


def test_warp_speed_keeps_idle_and_changes_core(seq):
    a, b = tm.active_segment(seq)
    w = tm.warp_speed(seq, 0.5, (a, b))
    np.testing.assert_allclose(w[:a], seq[:a])
    np.testing.assert_allclose(w[-(len(seq) - b - 1):], seq[b + 1:])
    m = (a + b) // 2
    # first half 2x slower, second half 2x faster
    assert len(w) - len(seq) == pytest.approx((m - a) - (b - m) / 2, abs=2)
    assert tm.warp_speed(seq, 1.0) is seq


def test_hold_pause_length_and_frozen_frames(seq):
    a, b = tm.active_segment(seq)
    h = tm.hold_frame(seq, (a, b))
    assert a <= h <= b
    out = tm.hold_pause(seq, 0.5, (a, b))
    n_hold = b - a + 1                                  # segment lasts 2x as long
    assert len(out) == len(seq) + n_hold
    np.testing.assert_allclose(out[h:h + n_hold + 1], np.repeat(seq[h:h + 1], n_hold + 1, axis=0))
    np.testing.assert_allclose(out[h + n_hold + 1:], seq[h + 1:])
    assert tm.hold_pause(seq, 1.5) is seq


def test_random_tempo_covers_segment_perturbations(seq):
    rng = np.random.default_rng(0)
    lengths = set()
    for _ in range(60):
        out = tm.random_tempo(seq, rng)
        assert np.isfinite(out).all() and len(out) >= 2
        assert tm.normalize(out, "uniform", CFG_T).shape == (32, seq.shape[1])
        lengths.add(len(out))
    assert any(n > len(seq) for n in lengths) and any(n < len(seq) for n in lengths)
    # same seed -> same augmentation (paired comparison across runs)
    a = tm.random_tempo(seq, np.random.default_rng(3))
    b = tm.random_tempo(seq, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)


def test_hold_applicable_only_when_slowing_down():
    assert tm.applicable("hold", 0.5) and tm.applicable("hold", 1.0)
    assert not tm.applicable("hold", 2.0)
    assert all(tm.applicable(k, 2.0) for k in ("global", "core", "warp"))


@pytest.mark.parametrize("kind", tm.PERTURBATIONS)
@pytest.mark.parametrize("factor", [0.25, 4.0])
def test_extreme_perturbations_normalise(seq, kind, factor):
    out = tm.perturb(seq, kind, factor)
    assert np.isfinite(out).all() and len(out) >= 2
    for method in ("uniform", "trim_uniform", "motion_aware"):
        assert tm.normalize(out, method, CFG_T).shape == (32, seq.shape[1])


def test_assign_hands_by_pose_wrist():
    pose = np.full((33, 4), np.nan, np.float32)
    pose[15, :2] = [0.7, 0.5]   # signer's left wrist (image right)
    pose[16, :2] = [0.3, 0.5]
    h_a = np.zeros((21, 3), np.float32); h_a[:, 0] = 0.31
    h_b = np.zeros((21, 3), np.float32); h_b[:, 0] = 0.69
    left, right = assign_hands([h_a, h_b], ["Left", "Right"], pose)
    assert left[0, 0] == pytest.approx(0.69) and right[0, 0] == pytest.approx(0.31)


def test_missing_hand_collapses_to_wrist():
    rng = np.random.default_rng(1)
    raw = make_sample(0, 0, rng, _class_params(8))
    raw["lhand"][:] = np.nan
    x = F.frame_features(raw)
    pts = x[:, :F.N_POINTS * 2].reshape(-1, F.N_POINTS, 2)
    np.testing.assert_allclose(pts[:, F.LHAND_SLICE].mean(1), pts[:, F.P_LWRIST], atol=1e-5)
    assert (x[:, -2] == 0).all()
