"""Experiment configuration: a YAML file merged over these defaults."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "paths": {
        # Folder that contains the WL-BISINDO videos (searched recursively).
        # "auto" searches /kaggle/input for a folder containing signer*_label*_sample* videos.
        "video_root": "auto",
        "work_dir": "outputs",          # index, features, models and results go here
        "results_name": "results",      # sub-folder of work_dir for this experiment's results
    },
    "extraction": {
        "pose_model": "lite",           # lite | full | heavy
        "num_workers": 4,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    },
    "features": {
        "hand_fill": "interpolate",     # interpolate | zero
    },
    "temporal": {
        "length": 32,                   # fixed number of timesteps fed to the classifier
        "methods": ["uniform", "trim_uniform", "motion_aware"],
        "motion_alpha": 0.3,            # 0 = pure motion density, 1 = uniform
        "motion_smooth": 5,             # moving-average window (frames) for motion energy
        "trim_quantiles": [0.02, 0.98], # cumulative-energy quantiles used for trimming
    },
    "split": {
        "protocol": "loso",             # loso (leave-one-signer-out) | si_signer4 (official SI split)
        "val_fraction": 0.15,           # stratified random split of the training signers
    },
    "model": {
        "type": "gru",                  # gru (PyTorch) | mlp (scikit-learn, simple baseline)
        "hidden": 128,
        "layers": 2,
        "dropout": 0.3,
        "epochs": 80,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 15,
        "train_speed_aug": False,       # extra comparator: uniform baseline trained with speed augmentation
        "train_full_aug": False,        # extra comparator: uniform + core/warp/hold augmentation, 0.25-4x
    },
    "stress": {
        "factors": [0.5, 0.75, 1.0, 1.25, 1.5],
        "types": ["global", "core"],
    },
    "seeds": [0, 1, 2],
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def work_path(cfg: dict, *parts: str) -> Path:
    p = Path(cfg["paths"]["work_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
