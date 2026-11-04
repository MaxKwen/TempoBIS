"""MediaPipe landmark extraction (MediaPipe Tasks API: PoseLandmarker + HandLandmarker).

Each video becomes one ``.npz`` file with
    pose  : (T, 33, 4)  x, y, z, visibility   (NaN when no person was detected)
    lhand : (T, 21, 3)  x, y, z of the signer's LEFT hand (NaN when missing)
    rhand : (T, 21, 3)  x, y, z of the signer's RIGHT hand
    fps, width, height
Coordinates are MediaPipe-normalised image coordinates ([0, 1] in x and y).

Hands are assigned to left/right by matching hand wrists to the pose wrists, which is
more reliable than the handedness label on non-mirrored video.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URLS = {
    "pose_lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}
POSE_L_WRIST, POSE_R_WRIST = 15, 16


def ensure_models(cache_dir: str | Path, pose_model: str = "lite") -> tuple[Path, Path]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for key in (f"pose_{pose_model}", "hand"):
        dst = cache / Path(MODEL_URLS[key]).name
        if not dst.exists():
            print(f"Downloading {dst.name} ...")
            urllib.request.urlretrieve(MODEL_URLS[key], dst)
        paths.append(dst)
    return paths[0], paths[1]


class LandmarkExtractor:
    """Wraps the two MediaPipe landmarkers. Create one per process; not thread-safe."""

    def __init__(self, pose_model_path, hand_model_path, min_det=0.5, min_track=0.5):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._vision = vision
        self._pose_opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=min_det, min_pose_presence_confidence=min_det,
            min_tracking_confidence=min_track)
        self._hand_opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=min_det, min_hand_presence_confidence=min_det,
            min_tracking_confidence=min_track)

    def extract(self, video_path: str) -> dict:
        import cv2
        import mediapipe as mp

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        pose_seq, lh_seq, rh_seq = [], [], []
        # VIDEO mode needs a fresh landmarker per video (timestamps must increase).
        with self._vision.PoseLandmarker.create_from_options(self._pose_opts) as pose_lm, \
                self._vision.HandLandmarker.create_from_options(self._hand_opts) as hand_lm:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = int(round(i * 1000.0 / fps))
                pose_res = pose_lm.detect_for_video(image, ts)
                hand_res = hand_lm.detect_for_video(image, ts)
                pose, lh, rh = _frame_arrays(pose_res, hand_res)
                pose_seq.append(pose)
                lh_seq.append(lh)
                rh_seq.append(rh)
                i += 1
        cap.release()
        if not pose_seq:
            raise IOError(f"No frames decoded from {video_path}")
        return {"pose": np.stack(pose_seq).astype(np.float32),
                "lhand": np.stack(lh_seq).astype(np.float32),
                "rhand": np.stack(rh_seq).astype(np.float32),
                "fps": np.float32(fps), "width": np.int32(width), "height": np.int32(height)}


def _frame_arrays(pose_res, hand_res):
    pose = np.full((33, 4), np.nan, np.float32)
    if pose_res.pose_landmarks:
        pose[:] = [[p.x, p.y, p.z, (p.visibility if p.visibility is not None else 1.0)]
                   for p in pose_res.pose_landmarks[0]]
    hands = [np.array([[p.x, p.y, p.z] for p in h], np.float32) for h in (hand_res.hand_landmarks or [])]
    labels = [h[0].category_name for h in (hand_res.handedness or [])]
    lh, rh = assign_hands(hands, labels, pose)
    return pose, lh, rh


def assign_hands(hands: list, labels: list, pose: np.ndarray):
    """Return (left, right) hand arrays (21, 3), NaN-filled when absent."""
    empty = np.full((21, 3), np.nan, np.float32)
    left, right = empty.copy(), empty.copy()
    if not hands:
        return left, right
    have_pose = not np.isnan(pose[[POSE_L_WRIST, POSE_R_WRIST], :2]).any()
    if have_pose:
        lw, rw = pose[POSE_L_WRIST, :2], pose[POSE_R_WRIST, :2]
        d = np.array([[np.linalg.norm(h[0, :2] - lw), np.linalg.norm(h[0, :2] - rw)] for h in hands])
        if len(hands) == 1:
            (left if d[0, 0] <= d[0, 1] else right)[:] = hands[0]
        else:
            # best of the two possible pairings for the first two hands
            if d[0, 0] + d[1, 1] <= d[0, 1] + d[1, 0]:
                left[:], right[:] = hands[0], hands[1]
            else:
                left[:], right[:] = hands[1], hands[0]
    else:
        # MediaPipe handedness assumes a mirrored (selfie) image; on normal video it is swapped.
        for h, lab in zip(hands[:2], labels[:2]):
            (right if lab == "Left" else left)[:] = h
    return left, right


# ---------------------------------------------------------------- batch extraction
_worker = None


def _init_worker(pose_path, hand_path, min_det, min_track):
    global _worker
    os.environ.setdefault("GLOG_minloglevel", "2")
    _worker = LandmarkExtractor(pose_path, hand_path, min_det, min_track)


def _run_one(args):
    video_path, out_path = args
    try:
        data = _worker.extract(video_path)
        np.savez_compressed(out_path, **data)
        return out_path, None
    except Exception as e:  # keep going; failures are reported at the end
        return out_path, repr(e)


def extract_dataset(index, out_dir, cache_dir, pose_model="lite", num_workers=4,
                    min_det=0.5, min_track=0.5, limit=None):
    """Extract every video in ``index`` (a DataFrame with video_id, path). Resumable."""
    from multiprocessing import get_context

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pose_path, hand_path = ensure_models(cache_dir, pose_model)
    todo = [(r.path, str(out / f"{r.video_id}.npz")) for r in index.itertuples()
            if not (out / f"{r.video_id}.npz").exists()]
    if limit:
        todo = todo[:limit]
    print(f"{len(index) - len(todo)} already extracted, {len(todo)} to go")
    failures = {}
    if not todo:
        return failures
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731
    ctx = get_context("spawn")
    with ctx.Pool(max(1, num_workers), initializer=_init_worker,
                  initargs=(str(pose_path), str(hand_path), min_det, min_track)) as pool:
        for out_path, err in tqdm(pool.imap_unordered(_run_one, todo), total=len(todo)):
            if err:
                failures[Path(out_path).stem] = err
    if failures:
        print(f"{len(failures)} videos failed; first: {next(iter(failures.items()))}")
    return failures


def load_landmarks(path: str | Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def detection_stats(raw: dict) -> dict:
    """Fraction of frames where pose / each hand was detected."""
    return {
        "frames": int(raw["pose"].shape[0]),
        "pose_rate": float(np.mean(~np.isnan(raw["pose"][:, 0, 0]))),
        "lhand_rate": float(np.mean(~np.isnan(raw["lhand"][:, 0, 0]))),
        "rhand_rate": float(np.mean(~np.isnan(raw["rhand"][:, 0, 0]))),
        "any_hand_rate": float(np.mean(~np.isnan(raw["lhand"][:, 0, 0]) | ~np.isnan(raw["rhand"][:, 0, 0]))),
    }
