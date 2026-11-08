"""Training + controlled speed stress test.

For every (fold, seed, temporal method):
  1. split the training signers' videos into train / validation (stratified by label,
     identical for all methods of the same fold and seed -> paired comparison)
  2. fixed-length sequences via temporal.normalize, fit the classifier
  3. evaluate on the held-out signer under every perturbation (kind x factor)
Rows are appended to results.csv as they finish, so an interrupted run resumes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from . import temporal as tm
from .features import frame_features
from .landmarks import load_landmarks
from .models import make_model


# ---------------------------------------------------------------- metrics
def macro_f1_present(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 averaged over the classes that occur in ``y_true``.

    A held-out signer may not have every gloss (WL-BISINDO signer 0 has 12 of 32). Averaging
    over all classes would add F1 = 0 for every absent class and cap the score at
    present/total. Predictions of absent classes still count as errors through the recall
    of the true class.
    """
    return float(f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true), zero_division=0))


# Comparators: uniform normalisation + training-time augmentation (2 extra copies per video).
#   uniform_speedaug - mild, constant speed change (global/core, 0.6-1.6x)
#   uniform_fullaug  - every stress-test perturbation type over the stress-test range (0.25-4x)
AUG_METHODS = {
    "uniform_speedaug": tm.random_speed,
    "uniform_fullaug": tm.random_tempo,
}


# ---------------------------------------------------------------- data
def load_sequences(index: pd.DataFrame, features_dir: str | Path, cfg: dict) -> tuple[dict, list]:
    seqs, missing = {}, []
    for vid in index["video_id"]:
        f = Path(features_dir) / f"{vid}.npz"
        if not f.exists():
            missing.append(vid)
            continue
        seqs[vid] = frame_features(load_landmarks(f), cfg["features"]["hand_fill"])
    return seqs, missing


def make_folds(index: pd.DataFrame, protocol: str) -> list[dict]:
    signers = sorted(index["signer"].unique())
    if protocol == "loso":
        test_signers = signers
    elif protocol == "si_signer4":
        if 4 not in signers:
            raise ValueError("si_signer4 needs signer 4 in the index")
        test_signers = [4]
    else:
        raise ValueError(f"unknown split protocol: {protocol}")
    folds = []
    for s in test_signers:
        folds.append({
            "fold": f"test_signer{s}",
            "test_signer": int(s),
            "train_ids": index.loc[index["signer"] != s, "video_id"].tolist(),
            "test_ids": index.loc[index["signer"] == s, "video_id"].tolist(),
        })
    return folds


def _stack(seqs, ids, method, cfg_t):
    return np.stack([tm.normalize(seqs[v], method, cfg_t) for v in ids])


# ---------------------------------------------------------------- main loop
def run(cfg: dict, index: pd.DataFrame, seqs: dict, out_dir: str | Path, verbose: bool = True) -> pd.DataFrame:
    out = Path(out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    with open(out / "config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)

    index = index[index["video_id"].isin(seqs)].reset_index(drop=True)
    classes = sorted(index["label"].unique())
    to_idx = {c: i for i, c in enumerate(classes)}
    label_of = dict(zip(index["video_id"], index["label"].map(to_idx)))
    cfg_t, cfg_m = cfg["temporal"], cfg["model"]
    methods = (list(cfg_t["methods"]) + (["uniform_speedaug"] if cfg_m.get("train_speed_aug") else [])
               + (["uniform_fullaug"] if cfg_m.get("train_full_aug") else []))
    kinds, factors = cfg["stress"]["types"], [float(f) for f in cfg["stress"]["factors"]]

    res_path = out / "results.csv"
    done = set()
    if res_path.exists():
        prev = pd.read_csv(res_path)
        done = set(zip(prev["fold"], prev["seed"], prev["method"]))

    folds = make_folds(index, cfg["split"]["protocol"])
    for fold in folds:
        test_ids = fold["test_ids"]
        y_test = np.array([label_of[v] for v in test_ids])
        # perturbed raw sequences are shared by all methods / seeds of this fold
        perturbed = {(k, f): {v: tm.perturb(seqs[v], k, f) for v in test_ids}
                     for k in kinds for f in factors if f != 1.0 and tm.applicable(k, f)}
        for seed in cfg["seeds"]:
            tr_ids, va_ids = train_test_split(
                fold["train_ids"], test_size=cfg["split"]["val_fraction"], random_state=seed,
                stratify=[label_of[v] for v in fold["train_ids"]])
            y_tr = np.array([label_of[v] for v in tr_ids])
            y_va = np.array([label_of[v] for v in va_ids])
            for method in methods:
                if (fold["fold"], seed, method) in done:
                    continue
                t0 = time.time()
                norm = "uniform" if method in AUG_METHODS else method
                X_tr, ytr = _stack(seqs, tr_ids, norm, cfg_t), y_tr
                if method in AUG_METHODS:
                    rng = np.random.default_rng(seed)
                    augment = AUG_METHODS[method]
                    aug = [tm.normalize(augment(seqs[v], rng), "uniform", cfg_t)
                           for _ in range(2) for v in tr_ids]
                    X_tr = np.concatenate([X_tr, np.stack(aug)])
                    ytr = np.concatenate([y_tr, y_tr, y_tr])
                X_va = _stack(seqs, va_ids, norm, cfg_t)
                model = make_model(cfg_m, len(classes), seed).fit(X_tr, ytr, X_va, y_va)

                rows = []
                clean_pred = model.predict_proba(_stack(seqs, test_ids, norm, cfg_t)).argmax(1)
                base = {"fold": fold["fold"], "test_signer": fold["test_signer"], "seed": seed,
                        "method": method, "model": cfg_m["type"], "n_train": len(X_tr),
                        "n_test": len(test_ids), "n_test_classes": int(len(np.unique(y_test))),
                        "n_classes": len(classes)}
                for k in kinds:
                    for f in factors:
                        if not tm.applicable(k, f):
                            continue
                        if f == 1.0:
                            pred = clean_pred
                        else:
                            Xp = np.stack([tm.normalize(perturbed[(k, f)][v], norm, cfg_t) for v in test_ids])
                            pred = model.predict_proba(Xp).argmax(1)
                        rows.append({**base, "perturbation": k, "factor": f,
                                     "accuracy": accuracy_score(y_test, pred),
                                     "macro_f1": macro_f1_present(y_test, pred)})
                tag = f"{fold['fold']}_seed{seed}_{method}"
                np.savez_compressed(out / "predictions" / f"{tag}.npz", video_id=np.array(test_ids),
                                    y_true=y_test, y_pred=clean_pred, classes=np.array(classes))
                model.save(out / "models" / (tag + (".pt" if cfg_m["type"] == "gru" else ".pkl")))
                df = pd.DataFrame(rows)
                df["train_seconds"] = round(time.time() - t0, 1)
                df.to_csv(res_path, mode="a", header=not res_path.exists(), index=False)
                if verbose:
                    clean = df[(df.factor == 1.0) & (df.perturbation == kinds[0])].iloc[0]
                    print(f"{tag:45s} clean acc={clean.accuracy:.3f} macroF1={clean.macro_f1:.3f} "
                          f"({df.train_seconds.iloc[0]}s)")
    return pd.read_csv(res_path)
