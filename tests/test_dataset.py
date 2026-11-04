from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tempobis.dataset import GLOSSES, audit, build_index, find_video_root
from tempobis.landmarks import _frame_arrays


def _write_video(path, n=12, size=(64, 48)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, size)
    for i in range(n):
        vw.write(np.full((size[1], size[0], 3), i * 10, np.uint8))
    vw.release()


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "input" / "wl-bisindo" / "videos"
    root.mkdir(parents=True)
    for s in (1, 2):
        for lab in (0, 7):
            _write_video(root / f"signer{s}_label{lab}_sample1.mp4", n=10 + s)
    (root / "notes.mp4").write_bytes(b"not a video")
    return tmp_path / "input"


def test_find_and_index(fake_root):
    root = find_video_root("auto", search_dirs=(str(fake_root),))
    assert root.name == "wl-bisindo"
    idx = build_index(root)
    assert len(idx) == 4 and set(idx.signer) == {1, 2}
    assert set(idx.gloss) == {GLOSSES[0], GLOSSES[7]}
    assert len(idx.attrs["unparsed"]) == 1


def test_audit_counts_frames(fake_root):
    idx = build_index(fake_root)
    table, summary = audit(idx, num_workers=2)
    assert summary["n_videos"] == 4 and summary["n_signers"] == 2
    assert sorted(table["decoded_frames"].unique()) == [11, 12]
    assert not summary["unreadable"]


def test_frame_arrays_from_mediapipe_like_results():
    lm = lambda x, y: SimpleNamespace(x=x, y=y, z=0.0, visibility=0.9)  # noqa: E731
    pose_lms = [lm(0.5, 0.5) for _ in range(33)]
    pose_lms[15], pose_lms[16] = lm(0.7, 0.6), lm(0.3, 0.6)
    pose_res = SimpleNamespace(pose_landmarks=[pose_lms])
    hand = lambda x: [lm(x, 0.6) for _ in range(21)]  # noqa: E731
    cat = lambda n: [SimpleNamespace(category_name=n)]  # noqa: E731
    hand_res = SimpleNamespace(hand_landmarks=[hand(0.3), hand(0.7)], handedness=[cat("Left"), cat("Right")])
    pose, lh, rh = _frame_arrays(pose_res, hand_res)
    assert pose.shape == (33, 4) and lh[0, 0] == pytest.approx(0.7) and rh[0, 0] == pytest.approx(0.3)
    empty = SimpleNamespace(pose_landmarks=[], hand_landmarks=[], handedness=[])
    pose, lh, rh = _frame_arrays(empty, empty)
    assert np.isnan(pose).all() and np.isnan(lh).all()
