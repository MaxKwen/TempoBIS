"""Generate notebooks/tempobis_kaggle.ipynb (keeps the notebook diff-free in code review)."""
import json
from pathlib import Path

CELLS = [
    ("md", """# TempoBIS on Kaggle — end-to-end runner

One notebook, one phase per section. Run a section, check its output, then continue.

**Before running**
1. *Add Input* → search **WL-BISINDO** (Kindy, Leonali & Lucky, 2025) and add it.
2. *Settings* → **Internet: On** (needed for `pip install` and the MediaPipe model files).
3. *Accelerator* → GPU (T4) is optional; it only speeds up training. Extraction runs on CPU.

Dataset license: **CC BY-NC 4.0** — fine for a non-commercial portfolio with attribution.
Do not re-upload the videos to GitHub."""),
    ("md", "## Setup — get the code"),
    ("code", """import os, sys, subprocess, glob

REPO_URL = ""   # e.g. "https://github.com/<your-username>/TempoBIS.git" once the repo is public

if REPO_URL:
    if not os.path.exists("/kaggle/working/TempoBIS"):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/kaggle/working/TempoBIS"], check=True)
    CODE = "/kaggle/working/TempoBIS"
else:
    # code uploaded as a Kaggle dataset (zip of this repo)
    hits = glob.glob("/kaggle/input/**/pyproject.toml", recursive=True)
    hits = [h for h in hits if "tempobis" in open(h).read().lower()]
    assert hits, "Set REPO_URL or add the TempoBIS code as an input dataset"
    import shutil
    shutil.copytree(os.path.dirname(hits[0]), "/kaggle/working/TempoBIS", dirs_exist_ok=True)
    CODE = "/kaggle/working/TempoBIS"

os.chdir(CODE)
!pip install -q -e . 2>&1 | tail -2
sys.path.insert(0, f"{CODE}/src")   # editable installs are only picked up by new kernels
print("code at", CODE)"""),
    ("code", """# Check: unit tests (includes the GRU test, which needs PyTorch — available on Kaggle)
!python -m pytest -q 2>&1 | tail -3"""),
    ("code", """import yaml
from tempobis.config import load_config, _merge
CONFIG = "configs/default.yaml"
cfg = load_config(CONFIG, overrides={"paths": {"work_dir": "/kaggle/working/outputs"}})
with open("configs/kaggle.yaml", "w") as f:
    yaml.safe_dump(cfg, f)
quick = _merge(cfg, yaml.safe_load(open("configs/quick.yaml")))  # same landmarks, separate results folder
with open("configs/kaggle_quick.yaml", "w") as f:
    yaml.safe_dump(quick, f)
CFG, QCFG = "configs/kaggle.yaml", "configs/kaggle_quick.yaml"
print(yaml.safe_dump(cfg["paths"]))"""),

    ("md", """## Phase 0 — Dataset audit
Goal: understand what is actually in the dataset before touching any model.
Check: number of videos / glosses / signers, videos per signer, fps, frame counts, broken or duplicate files."""),
    ("code", "!python -m tempobis audit --config $CFG"),
    ("code", """import pandas as pd
audit = pd.read_csv("/kaggle/working/outputs/audit/audit_videos.csv")
display(pd.crosstab(audit.gloss, audit.signer).style.background_gradient(axis=None))
audit[["decoded_frames", "duration_s", "fps"]].describe()"""),
    ("md", """**Write down** (for the README / report): totals, videos per signer, fps, frame-count range, anything broken.
If a signer has far fewer videos, note it — it affects the leave-one-signer-out folds."""),

    ("md", """## Phase 1 — MediaPipe pilot (10 videos)
Goal: landmarks can be extracted and look right. Check: detection rates and one visual overlay."""),
    ("code", "!python -m tempobis pilot --config $CFG --n 10"),
    ("code", """import numpy as np, matplotlib.pyplot as plt, cv2
from tempobis.landmarks import load_landmarks
from tempobis.features import frame_features, wrist_heights
from tempobis.temporal import motion_energy, active_segment, trim_bounds

pilot = sorted(glob.glob("/kaggle/working/outputs/pilot/*.npz"))
vid = os.path.basename(pilot[0])[:-4]
raw = load_landmarks(pilot[0])
row = audit[audit.video_id == vid].iloc[0]
cap = cv2.VideoCapture(row.path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, t in zip(axes, np.linspace(0, len(raw["pose"]) - 1, 4).astype(int)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, fr = cap.read()
    h, w = fr.shape[:2]; ax.imshow(fr[..., ::-1])
    for key, c in (("pose", "y"), ("lhand", "c"), ("rhand", "m")):
        p = raw[key][t]; ax.scatter(p[:, 0] * w, p[:, 1] * h, s=6, c=c)
    ax.set_title(f"{vid} frame {t}"); ax.axis("off")
plt.show()

x = frame_features(raw); e = motion_energy(x); a, b = active_segment(x); lo, hi = trim_bounds(x)
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(e, label="motion energy"); ax.axvspan(a, b, alpha=0.15, label="active (wrist height)")
ax.axvline(lo, ls="--", c="k"); ax.axvline(hi, ls="--", c="k", label="trim bounds")
ax.legend(); ax.set_xlabel("frame"); plt.show()"""),
    ("md", """**Check before continuing:** hand points sit on the hands; left/right are not swapped;
the active segment and trim bounds cover the sign. If hands are often missed, try `pose_model: full`
or lower `min_detection_confidence` in the config before extracting everything."""),

    ("md", "## Phase 2 — Extract the whole dataset (resumable; ~30–60 min on 4 CPU workers)"),
    ("code", "!python -m tempobis extract --config $CFG"),
    ("code", """st = pd.read_csv("/kaggle/working/outputs/landmark_stats.csv")
print(len(st), "videos extracted")
st[["frames", "pose_rate", "lhand_rate", "rhand_rate", "any_hand_rate"]].describe()"""),
    ("md", "**Check:** every video extracted; note videos with very low `any_hand_rate` (used later in error analysis)."),

    ("md", """## Phase 3 — Quick baseline sanity run
Official signer-independent split (test signer 4), one seed, fewer epochs. Goal: the model learns
something sensible without leakage before spending GPU time on the full experiment."""),
    ("code", """!python -m tempobis train --config $QCFG 2>&1 | tail -30"""),
    ("md", """Results go to `outputs/results_quick/`. If clean macro-F1 is near chance (1/32 ≈ 0.03),
stop and debug (labels, features) before Phase 4–5."""),

    ("md", """## Phases 4–5 — Full experiment: baseline vs motion-aware under the speed stress test
5 leave-one-signer-out folds × 3 seeds × 4 methods (uniform, trim+uniform, motion-aware,
uniform+speed-aug), each evaluated at 5 speed factors × 2 perturbation types. Resumable: rerun the
cell if the session restarts (copy `outputs/results` to a dataset to keep it between sessions)."""),
    ("code", "!python -m tempobis train --config $CFG 2>&1 | tail -80"),
    ("code", """from IPython.display import Markdown, Image
display(Markdown(open("/kaggle/working/outputs/results/summary.md").read()))
display(Image("/kaggle/working/outputs/results/figures/robustness_curves.png"))"""),

    ("md", "## Phase 6 — Error analysis"),
    ("code", "!python -m tempobis analyze --config $CFG"),
    ("code", """display(Image("/kaggle/working/outputs/results/figures/confusion_uniform.png"))
display(Image("/kaggle/working/outputs/results/figures/confusion_motion_aware.png"))"""),

    ("md", """## Save outputs
Download `tempobis_results.zip` (Output tab) and send `summary.md`, `error_analysis_*.md`,
`audit_summary.json` and the figures back for the write-up. Models for the demo are in `results/models/`."""),
    ("code", """!cd /kaggle/working/outputs && zip -qr /kaggle/working/tempobis_results.zip results audit landmark_stats.csv -x "results/models/*"
!cd /kaggle/working/outputs && zip -qr /kaggle/working/tempobis_models.zip results/models results/config_used.json
!ls -lh /kaggle/working/*.zip"""),
]


def build(path: Path):
    cells = []
    for kind, src in CELLS:
        lines = [l + "\n" for l in src.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": lines})
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}}}
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "notebooks" / "tempobis_kaggle.ipynb"
    build(out)
    print("wrote", out)
