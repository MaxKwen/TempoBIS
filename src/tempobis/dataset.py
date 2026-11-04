"""Locate, index and audit WL-BISINDO videos.

WL-BISINDO file names follow ``signer{S}_label{L}_sample{N}.mp4`` (see the official
split metadata in github.com/AceKinnn/WL-BISINDO). Signer, label and sample are parsed
from the file name, so the index does not depend on how the Kaggle folders are nested.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Label -> gloss, from the WL-BISINDO README (Kindy, Leonali & Lucky, 2025).
GLOSSES = [
    "Air", "Belajar", "Cari", "Hari", "Ingat", "Lagi", "Maaf", "Makan",
    "Motor", "Saya", "Terima kasih", "Tuli", "Apa", "Siapa", "Kapan", "Di mana",
    "Mengapa", "Bagaimana", "Merah", "Kuning", "Hijau", "Hitam", "Dengar", "Berangkat",
    "Datang", "Teman", "Keluarga", "Rumah", "Pagi", "Siang", "Sore", "Malam",
]

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
NAME_RE = re.compile(r"signer(\d+)_label(\d+)_sample(\d+)", re.IGNORECASE)


def find_video_root(hint: str = "auto", search_dirs=("/kaggle/input",)) -> Path:
    """Return the folder holding the videos. ``hint`` may be a path or "auto"."""
    if hint and hint != "auto":
        p = Path(hint)
        if not p.exists():
            raise FileNotFoundError(f"video_root does not exist: {p}")
        return p
    for base in map(Path, search_dirs):
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if f.suffix.lower() in VIDEO_EXTS and NAME_RE.search(f.stem):
                # climb to the dataset's top folder (direct child of the search dir)
                rel = f.relative_to(base)
                return base / rel.parts[0] if len(rel.parts) > 1 else base
    raise FileNotFoundError(
        "No WL-BISINDO videos found. Add the dataset to the notebook (Kaggle: 'Add Input') "
        "or set paths.video_root in the config."
    )


def build_index(video_root: str | Path) -> pd.DataFrame:
    """One row per video, parsed from file names. Duplicated video ids are kept and flagged."""
    rows, unparsed = [], []
    for f in sorted(Path(video_root).rglob("*")):
        if f.suffix.lower() not in VIDEO_EXTS:
            continue
        m = NAME_RE.search(f.stem)
        if not m:
            unparsed.append(str(f))
            continue
        signer, label, sample = map(int, m.groups())
        rows.append({
            "video_id": f"signer{signer}_label{label}_sample{sample}",
            "path": str(f),
            "signer": signer,
            "label": label,
            "gloss": GLOSSES[label] if 0 <= label < len(GLOSSES) else f"label{label}",
            "sample": sample,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No parsable videos under {video_root}")
    df["dup_id"] = df.duplicated("video_id", keep="first")
    df.attrs["unparsed"] = unparsed
    return df


def _probe(path: str) -> dict:
    import cv2

    cap = cv2.VideoCapture(path)
    info = {"readable": cap.isOpened(), "fps": np.nan, "frames": 0, "width": 0, "height": 0,
            "decoded_frames": 0}
    if cap.isOpened():
        info["fps"] = float(cap.get(cv2.CAP_PROP_FPS) or np.nan)
        info["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        # Container frame counts can lie; decode to count real frames.
        n = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            n += 1
        info["decoded_frames"] = n
    cap.release()
    info["duration_s"] = info["decoded_frames"] / info["fps"] if info["fps"] and info["fps"] > 0 else np.nan
    with open(path, "rb") as fh:
        info["md5_head"] = hashlib.md5(fh.read(1 << 20)).hexdigest()
    info["size_mb"] = Path(path).stat().st_size / 1e6
    return info


def audit(index: pd.DataFrame, num_workers: int = 4) -> tuple[pd.DataFrame, dict]:
    """Probe every video and return (per-video table, summary dict)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        probes = list(ex.map(_probe, index["path"].tolist()))
    table = pd.concat([index.reset_index(drop=True), pd.DataFrame(probes)], axis=1)
    table["content_dup"] = table.duplicated("md5_head", keep=False)

    per_label_signer = pd.crosstab(table["label"], table["signer"])
    summary = {
        "n_videos": int(len(table)),
        "n_labels": int(table["label"].nunique()),
        "n_signers": int(table["signer"].nunique()),
        "signers": sorted(int(s) for s in table["signer"].unique()),
        "videos_per_signer": {int(k): int(v) for k, v in table["signer"].value_counts().sort_index().items()},
        "videos_per_label_min_max": [int(table["label"].value_counts().min()), int(table["label"].value_counts().max())],
        "unreadable": table.loc[~table["readable"] | (table["decoded_frames"] == 0), "video_id"].tolist(),
        "duplicate_ids": table.loc[table["dup_id"], "video_id"].tolist(),
        "content_duplicates": table.loc[table["content_dup"], "video_id"].tolist(),
        "unparsed_files": index.attrs.get("unparsed", []),
        "fps_values": {str(k): int(v) for k, v in table["fps"].round(2).value_counts().items()},
        "resolutions": {f"{w}x{h}": int(c) for (w, h), c in table.groupby(["width", "height"]).size().items()},
        "frames": table["decoded_frames"].describe().round(1).to_dict(),
        "duration_s": table["duration_s"].describe().round(2).to_dict(),
        "empty_label_signer_cells": int((per_label_signer == 0).sum().sum()),
    }
    return table, summary


def save_audit(table: pd.DataFrame, summary: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "audit_videos.csv", index=False)
    pd.crosstab(table["gloss"], table["signer"]).to_csv(out / "audit_label_by_signer.csv")
    with open(out / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
