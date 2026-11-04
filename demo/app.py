"""TempoBIS demo: upload one isolated BISINDO sign video -> top-5 glosses.

Run locally:   python demo/app.py --model path/to/test_signer4_seed0_motion_aware.pt
Hugging Face Spaces: put the model file in demo/model_motion_aware.pt and config_used.json next to it;
app.py picks them up automatically (see docs/deploy_demo.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tempobis import temporal as tm  # noqa: E402
from tempobis.dataset import GLOSSES  # noqa: E402
from tempobis.features import frame_features  # noqa: E402
from tempobis.landmarks import LandmarkExtractor, ensure_models  # noqa: E402
from tempobis.models import load_model  # noqa: E402

METHODS = ("motion_aware", "trim_uniform", "uniform_speedaug", "uniform")


def method_from_name(path: str) -> str:
    name = Path(path).stem
    for m in METHODS:
        if name.endswith(m):
            return "uniform" if m == "uniform_speedaug" else m
    return "motion_aware"


class Predictor:
    def __init__(self, model_path: str, config_path: str | None = None):
        self.model = load_model(model_path)
        cfg_path = config_path or Path(model_path).parent.parent / "config_used.json"
        if not Path(cfg_path).exists():
            cfg_path = HERE / "config_used.json"
        with open(cfg_path) as f:
            self.cfg = json.load(f)
        self.method = method_from_name(model_path)
        pose_p, hand_p = ensure_models(Path(tempfile.gettempdir()) / "tempobis_mp",
                                       self.cfg["extraction"]["pose_model"])
        self.extractor = LandmarkExtractor(pose_p, hand_p)

    def __call__(self, video_path: str):
        raw = self.extractor.extract(video_path)
        seq = frame_features(raw, self.cfg["features"]["hand_fill"])
        x = tm.normalize(seq, self.method, self.cfg["temporal"])[None]
        probs = self.model.predict_proba(x)[0]
        top = {GLOSSES[i] if i < len(GLOSSES) else str(i): float(probs[i]) for i in np.argsort(probs)[::-1][:5]}
        hand_rate = float(np.mean(~np.isnan(raw["lhand"][:, 0, 0]) | ~np.isnan(raw["rhand"][:, 0, 0])))
        return top, self._plot(seq), f"{len(seq)} frames · hands detected in {hand_rate:.0%} of frames · " \
                                     f"temporal method: {self.method}"

    def _plot(self, seq):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        e = tm.motion_energy(seq, self.cfg["temporal"]["motion_smooth"])
        T, alpha = self.cfg["temporal"]["length"], self.cfg["temporal"]["motion_alpha"]
        uni = np.linspace(0, len(seq) - 1, T)
        if e.sum() > 0:
            dens = (1 - alpha) * e / e.sum() + alpha / len(e)
            ma = tm._positions_from_density(dens, T)
        else:
            ma = uni
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.plot(np.arange(len(e)) + 0.5, e, color="#444", lw=1.5, label="hand motion energy")
        top = e.max() if e.max() > 0 else 1
        ax.vlines(uni, top * 1.05, top * 1.15, color="#1f77b4", lw=1, label="uniform timesteps")
        ax.vlines(ma, top * 1.2, top * 1.3, color="#d62728", lw=1, label="motion-aware timesteps")
        ax.set_xlabel("frame"); ax.set_yticks([]); ax.legend(fontsize=7, loc="upper right", ncol=3)
        ax.set_ylim(0, top * 1.45); fig.tight_layout()
        return fig


def build_ui(pred: Predictor):
    import gradio as gr

    with gr.Blocks(title="TempoBIS") as ui:
        gr.Markdown("# TempoBIS — isolated BISINDO sign recognition\n"
                    "Upload a short video of **one** sign (one of the 32 WL-BISINDO glosses). "
                    "Research demo; trained on 5 signers from Banten, so expect errors on other "
                    "signers, regional variants and camera setups.")
        with gr.Row():
            vid = gr.Video(label="Sign video", sources=["upload", "webcam"])
            with gr.Column():
                out = gr.Label(num_top_classes=5, label="Prediction")
                info = gr.Markdown()
        plot = gr.Plot(label="Where each method places its timesteps")
        gr.Button("Recognize", variant="primary").click(pred, vid, [out, plot, info])
        gr.Markdown("Glosses: " + ", ".join(GLOSSES))
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("TEMPOBIS_MODEL", str(HERE / "model_motion_aware.pt")))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    build_ui(Predictor(args.model, args.config)).launch()
