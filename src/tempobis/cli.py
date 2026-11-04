"""Command line entry point:  python -m tempobis <command> [--config configs/default.yaml]

  audit      index + probe every video          -> outputs/audit/
  pilot      extract landmarks for N videos      -> outputs/pilot/ (+ detection stats)
  extract    extract landmarks for all videos    -> outputs/landmarks/
  train      training + speed stress test        -> outputs/results/
  report     tables and figures from results     -> outputs/results/summary.md
  analyze    per-class errors, confusions        -> outputs/results/error_analysis_*.md
  smoke      whole pipeline on synthetic data    -> outputs_smoke/ (code check only)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config, work_path


def _results_dir(cfg):
    return work_path(cfg, cfg["paths"].get("results_name", "results"), "x").parent


def _index(cfg, refresh=False):
    from .dataset import build_index, find_video_root

    p = work_path(cfg, "index.csv")
    if p.exists() and not refresh:
        return pd.read_csv(p)
    root = find_video_root(cfg["paths"]["video_root"])
    print(f"Video root: {root}")
    idx = build_index(root)
    idx.to_csv(p, index=False)
    return idx


def cmd_audit(cfg, args):
    from .dataset import audit, save_audit

    idx = _index(cfg, refresh=True)
    table, summary = audit(idx, cfg["extraction"]["num_workers"])
    save_audit(table, summary, work_path(cfg, "audit", "x").parent)
    print(json.dumps(summary, indent=2, default=str))


def cmd_pilot(cfg, args):
    from .landmarks import detection_stats, extract_dataset, load_landmarks

    idx = _index(cfg)
    # one video per label, spread over signers, so the pilot covers different signs
    pilot = idx.groupby("label").sample(1, random_state=0).head(args.n)
    out = work_path(cfg, "pilot", "x").parent
    fails = extract_dataset(pilot, out, work_path(cfg, "mp_models", "x").parent,
                            cfg["extraction"]["pose_model"], min(cfg["extraction"]["num_workers"], len(pilot)))
    stats = []
    for r in pilot.itertuples():
        f = out / f"{r.video_id}.npz"
        if f.exists():
            stats.append({"video_id": r.video_id, "gloss": r.gloss, **detection_stats(load_landmarks(f))})
    df = pd.DataFrame(stats)
    df.to_csv(out / "pilot_detection.csv", index=False)
    print(df.to_string(index=False))
    if fails:
        print("Failures:", fails)


def cmd_extract(cfg, args):
    from .landmarks import detection_stats, extract_dataset, load_landmarks

    idx = _index(cfg)
    idx = idx[~idx["dup_id"]] if "dup_id" in idx else idx
    out = work_path(cfg, "landmarks", "x").parent
    fails = extract_dataset(idx, out, work_path(cfg, "mp_models", "x").parent, cfg["extraction"]["pose_model"],
                            cfg["extraction"]["num_workers"], cfg["extraction"]["min_detection_confidence"],
                            cfg["extraction"]["min_tracking_confidence"], limit=args.limit)
    stats = [{"video_id": f.stem, **detection_stats(load_landmarks(f))} for f in sorted(out.glob("*.npz"))]
    pd.DataFrame(stats).to_csv(work_path(cfg, "landmark_stats.csv"), index=False)
    if fails:
        with open(work_path(cfg, "extraction_failures.json"), "w") as f:
            json.dump(fails, f, indent=2)


def cmd_train(cfg, args):
    from .experiment import load_sequences, run

    idx = _index(cfg)
    idx = idx[~idx["dup_id"]] if "dup_id" in idx else idx
    seqs, missing = load_sequences(idx, work_path(cfg, "landmarks", "x").parent, cfg)
    print(f"{len(seqs)} sequences loaded, {len(missing)} missing")
    run(cfg, idx, seqs, _results_dir(cfg))
    cmd_report(cfg, args)


def cmd_report(cfg, args, synthetic=False):
    from .dataset import GLOSSES
    from .report import write_report

    print(write_report(_results_dir(cfg), None if synthetic else GLOSSES, synthetic))


def cmd_analyze(cfg, args):
    from .dataset import GLOSSES
    from .report import error_analysis

    res = _results_dir(cfg)
    for m in ("uniform", "motion_aware"):
        print(error_analysis(res, m, GLOSSES, work_path(cfg, "landmark_stats.csv")), "\n")


def cmd_smoke(cfg, args):
    from .experiment import load_sequences, run
    from .synthetic import make_dataset

    cfg["paths"]["work_dir"] = args.out
    idx = make_dataset(work_path(cfg, "landmarks", "x").parent, n_classes=8, n_signers=4, n_samples=6)
    seqs, _ = load_sequences(idx, work_path(cfg, "landmarks", "x").parent, cfg)
    run(cfg, idx, seqs, _results_dir(cfg))
    cmd_report(cfg, args, synthetic=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tempobis", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["audit", "pilot", "extract", "train", "report", "analyze", "smoke"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=10, help="pilot: number of videos")
    ap.add_argument("--limit", type=int, default=None, help="extract: only the first N pending videos")
    ap.add_argument("--model", default=None, help="override model.type (gru|mlp)")
    ap.add_argument("--out", default="outputs_smoke", help="smoke: output folder")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["type"] = args.model
    {"audit": cmd_audit, "pilot": cmd_pilot, "extract": cmd_extract, "train": cmd_train,
     "report": cmd_report, "analyze": cmd_analyze, "smoke": cmd_smoke}[args.command](cfg, args)


if __name__ == "__main__":
    main()
