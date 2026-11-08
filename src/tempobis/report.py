"""Aggregate results.csv into tables and figures (no numbers are typed by hand)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METHOD_LABELS = {
    "uniform": "Uniform (baseline)",
    "trim_uniform": "Trim + uniform (ablation)",
    "motion_aware": "Motion-aware (proposed)",
    "uniform_speedaug": "Uniform + speed aug",
    "uniform_fullaug": "Uniform + full tempo aug",
}
METRIC = "macro_f1"


def _fmt(m, s):
    return f"{m:.3f} ± {s:.3f}" if np.isfinite(s) else f"{m:.3f}"


def per_run_robustness(res: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    """One row per (fold, seed, method, perturbation): clean, mean perturbed, drop, worst."""
    rows = []
    for key, g in res.groupby(["fold", "seed", "method", "perturbation"]):
        clean = g.loc[g.factor == 1.0, metric].mean()
        pert = g.loc[g.factor != 1.0, metric]
        rows.append(dict(zip(["fold", "seed", "method", "perturbation"], key),
                         clean=clean, perturbed_mean=pert.mean(), drop=clean - pert.mean(),
                         worst=pert.min()))
    return pd.DataFrame(rows)


def curve_table(res: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    """mean ± std over folds (seeds averaged inside each fold first)."""
    by_fold = res.groupby(["method", "perturbation", "factor", "fold"])[metric].mean().reset_index()
    agg = by_fold.groupby(["method", "perturbation", "factor"])[metric].agg(["mean", "std"]).reset_index()
    return agg


def paired_test(rob: pd.DataFrame, a: str, b: str, perturbation: str, col: str = "drop") -> dict:
    """Paired comparison of method a vs b over matching (fold, seed) runs."""
    pa = rob[(rob.method == a) & (rob.perturbation == perturbation)].set_index(["fold", "seed"])[col]
    pb = rob[(rob.method == b) & (rob.perturbation == perturbation)].set_index(["fold", "seed"])[col]
    common = pa.index.intersection(pb.index)
    d = (pa.loc[common] - pb.loc[common]).to_numpy()
    out = {"n_pairs": int(len(d)), "mean_diff": float(np.mean(d)) if len(d) else np.nan, "p_wilcoxon": np.nan}
    if len(d) >= 6 and np.any(d != 0):
        from scipy.stats import wilcoxon

        out["p_wilcoxon"] = float(wilcoxon(d).pvalue)
    return out


def plot_curves(res: pd.DataFrame, path: Path, metric: str = METRIC, title_suffix: str = ""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = curve_table(res, metric)
    kinds = list(dict.fromkeys(res["perturbation"]))
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.2 * len(kinds), 4), sharey=True, squeeze=False)
    names = {"global": "Whole clip speed change", "core": "Signing-segment-only speed change",
             "warp": "Uneven tempo inside the sign", "hold": "Pause (hold) at the sign's peak"}
    wide = res["factor"].max() / res["factor"].min() > 4
    for ax, k in zip(axes[0], kinds):
        for m in dict.fromkeys(res["method"]):
            g = agg[(agg.method == m) & (agg.perturbation == k)].sort_values("factor")
            ax.plot(g.factor, g["mean"], marker="o", label=METHOD_LABELS.get(m, m))
            if g["std"].notna().any():
                ax.fill_between(g.factor, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.15)
        ax.axvline(1.0, color="grey", lw=0.8, ls=":")
        if wide:
            fs = sorted(res["factor"].unique())
            ax.set_xscale("log"); ax.set_xticks(fs); ax.set_xticklabels([f"{f:g}" for f in fs])
            ax.minorticks_off()
        ax.set_title(names.get(k, k))
        ax.set_xlabel("Speed factor (>1 = faster)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(metric.replace("_", "-").title())
    axes[0][-1].legend(fontsize=8, loc="lower center")
    fig.suptitle(f"Speed stress test{title_suffix}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(pred_dir: Path, method: str, path: Path, glosses: list[str] | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    files = sorted(pred_dir.glob(f"*_{method}.npz"))
    if not files:
        return None
    yt, yp, classes = [], [], None
    for f in files:
        with np.load(f) as z:
            yt.append(z["y_true"]); yp.append(z["y_pred"]); classes = z["classes"]
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    cm = confusion_matrix(yt, yp, labels=range(len(classes)), normalize="true")
    labels = [glosses[c] if glosses and c < len(glosses) else str(c) for c in classes]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.32), max(4.5, n * 0.3)))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion (row-normalised), {METHOD_LABELS.get(method, method)}, speed 1.0x", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def coverage_section(res: pd.DataFrame, metric: str = METRIC) -> list[str]:
    """Flag folds whose held-out signer lacks some classes; clean scores with and without them."""
    if "n_test_classes" not in res or "n_classes" not in res:
        return ["> Results predate class-coverage tracking: macro-F1 may average over classes",
                "> absent from the held-out signer (F1 = 0 each). Re-run training to fix.", ""]
    cov = res.groupby("fold")[["n_test_classes", "n_classes"]].first()
    reduced = cov[cov.n_test_classes < cov.n_classes]
    if reduced.empty:
        return []
    lines = ["Macro-F1 is averaged over the classes present in each held-out signer. Reduced-coverage folds: "
             + ", ".join(f"{f} ({r.n_test_classes}/{r.n_classes} classes)" for f, r in reduced.iterrows()) + ".",
             "", "Clean macro-F1 with and without those folds:", "",
             "| Method | all folds | full-coverage folds only |", "|---|---:|---:|"]
    clean = res[res.factor == 1.0].groupby(["method", "fold"])[metric].mean().reset_index()
    for m in dict.fromkeys(res["method"]):
        c = clean[clean.method == m]
        full = c[~c.fold.isin(reduced.index)][metric]
        lines.append(f"| {METHOD_LABELS.get(m, m)} | {_fmt(c[metric].mean(), c[metric].std())} | "
                     f"{_fmt(full.mean(), full.std())} |")
    return lines + [""]


def write_report(res_dir: str | Path, glosses: list[str] | None = None, synthetic: bool = False) -> str:
    res_dir = Path(res_dir)
    res = pd.read_csv(res_dir / "results.csv")
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    warn = " (SYNTHETIC DATA - code check only)" if synthetic else ""

    plot_curves(res, fig_dir / "robustness_curves.png", title_suffix=warn)
    for m in res["method"].unique():
        plot_confusion(res_dir / "predictions", m, fig_dir / f"confusion_{m}.png", glosses)

    rob = per_run_robustness(res)
    rob.to_csv(res_dir / "robustness_per_run.csv", index=False)
    agg = curve_table(res)
    agg.to_csv(res_dir / "curve_table.csv", index=False)

    lines = [f"# TempoBIS results{warn}", ""]
    if synthetic:
        lines += ["> These numbers come from synthetic landmarks and only show that the code runs.",
                  "> They are NOT BISINDO results and must not be reported.", ""]
    n_folds, n_seeds = res["fold"].nunique(), res["seed"].nunique()
    lines += [f"Model: `{res['model'].iloc[0]}` · folds: {n_folds} ({', '.join(sorted(res['fold'].unique()))}) "
              f"· seeds: {n_seeds} · metric: macro-F1 on the held-out signer, mean ± std over folds.", ""]
    lines += coverage_section(res)

    methods = list(dict.fromkeys(res["method"]))
    for k in dict.fromkeys(res["perturbation"]):
        factors = sorted(res["factor"].unique())
        lines += [f"## Perturbation: {k}", "",
                  "| Method | " + " | ".join(f"{f:g}×" for f in factors) + " |",
                  "|---|" + "---:|" * len(factors)]
        for m in methods:
            cells = []
            for f in factors:
                r = agg[(agg.method == m) & (agg.perturbation == k) & (agg.factor == f)]
                cells.append(_fmt(r["mean"].iloc[0], r["std"].iloc[0]) if len(r) else "–")
            lines.append(f"| {METHOD_LABELS.get(m, m)} | " + " | ".join(cells) + " |")
        lines += ["", "| Method | clean | mean perturbed | drop (clean − perturbed) | worst |",
                  "|---|---:|---:|---:|---:|"]
        for m in methods:
            r = rob[(rob.method == m) & (rob.perturbation == k)]
            f_ = r.groupby("fold")[["clean", "perturbed_mean", "drop", "worst"]].mean()
            lines.append(f"| {METHOD_LABELS.get(m, m)} | " + " | ".join(
                _fmt(f_[c].mean(), f_[c].std()) for c in ["clean", "perturbed_mean", "drop", "worst"]) + " |")
        if "uniform" in methods:
            lines += ["", "Paired vs uniform baseline over (fold, seed) runs "
                          "(negative drop difference = smaller drop than baseline):", ""]
            for m in methods:
                if m == "uniform":
                    continue
                t_drop = paired_test(rob, m, "uniform", k, "drop")
                t_clean = paired_test(rob, m, "uniform", k, "clean")
                lines.append(f"- {METHOD_LABELS.get(m, m)}: Δdrop = {t_drop['mean_diff']:+.3f} "
                             f"(Wilcoxon p = {t_drop['p_wilcoxon']:.3g}), Δclean = {t_clean['mean_diff']:+.3f} "
                             f"(p = {t_clean['p_wilcoxon']:.3g}), n = {t_drop['n_pairs']}")
        lines.append("")
    lines += ["## Figures", "", "![robustness](figures/robustness_curves.png)", ""]
    text = "\n".join(lines)
    (res_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def error_analysis(res_dir: str | Path, method: str, glosses: list[str] | None = None,
                   landmark_stats: str | Path | None = None, top_k: int = 10) -> str:
    """Per-class F1, most confused pairs and (optionally) error rate vs hand-detection rate."""
    from sklearn.metrics import classification_report

    res_dir = Path(res_dir)
    frames = []
    for f in sorted((res_dir / "predictions").glob(f"*_{method}.npz")):
        with np.load(f) as z:
            classes = z["classes"]
            frames.append(pd.DataFrame({"video_id": z["video_id"], "y_true": z["y_true"], "y_pred": z["y_pred"],
                                        "run": f.stem}))
    if not frames:
        return f"No predictions for method {method}"
    df = pd.concat(frames, ignore_index=True)
    name = lambda i: glosses[classes[i]] if glosses else str(classes[i])  # noqa: E731
    rep = classification_report(df.y_true, df.y_pred, labels=range(len(classes)),
                                target_names=[name(i) for i in range(len(classes))],
                                output_dict=True, zero_division=0)
    per_class = (pd.DataFrame(rep).T.iloc[:len(classes)][["precision", "recall", "f1-score", "support"]]
                 .sort_values("f1-score"))
    per_class.to_csv(res_dir / f"per_class_{method}.csv")
    wrong = df[df.y_true != df.y_pred]
    pairs = (wrong.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False).head(top_k)
             .reset_index(name="count"))
    lines = [f"# Error analysis: {METHOD_LABELS.get(method, method)} (speed 1.0x, all folds/seeds)", "",
             "## Hardest classes", "", "| Gloss | precision | recall | F1 |", "|---|---:|---:|---:|"]
    for g, r in per_class.head(top_k).iterrows():
        lines.append(f"| {g} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1-score']:.2f} |")
    lines += ["", "## Most confused pairs (true → predicted)", "", "| True | Predicted | count |", "|---|---|---:|"]
    for r in pairs.itertuples():
        lines.append(f"| {name(r.y_true)} | {name(r.y_pred)} | {r.count} |")
    if landmark_stats and Path(landmark_stats).exists():
        st = pd.read_csv(landmark_stats)
        per_vid = df.assign(correct=df.y_true == df.y_pred).groupby("video_id")["correct"].mean().reset_index()
        m = per_vid.merge(st, on="video_id", how="left")
        m["hand_bin"] = pd.cut(m["any_hand_rate"], [0, 0.5, 0.8, 0.95, 1.0001], include_lowest=True)
        tab = m.groupby("hand_bin", observed=True)["correct"].agg(["mean", "size"])
        lines += ["", "## Accuracy vs hand-detection rate", "",
                  "| frames with a hand detected | accuracy | videos |", "|---|---:|---:|"]
        for b, r in tab.iterrows():
            lines.append(f"| {b} | {r['mean']:.3f} | {int(r['size'])} |")
    text = "\n".join(lines)
    (res_dir / f"error_analysis_{method}.md").write_text(text, encoding="utf-8")
    return text
