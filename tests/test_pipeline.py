import importlib.util

import pandas as pd
import pytest

from tempobis.config import load_config
from tempobis.experiment import load_sequences, make_folds, run
from tempobis.report import write_report
from tempobis.synthetic import make_dataset


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    idx = make_dataset(d / "landmarks", n_classes=4, n_signers=3, n_samples=4)
    return d, idx


def _cfg(model="mlp", **kw):
    return load_config(overrides={"model": {"type": model, "epochs": 3, "patience": 2, **kw},
                                  "seeds": [0], "temporal": {"length": 16}})


def test_loso_folds_have_no_signer_overlap(tiny):
    _, idx = tiny
    folds = make_folds(idx, "loso")
    assert len(folds) == idx.signer.nunique()
    for f in folds:
        train_signers = set(idx[idx.video_id.isin(f["train_ids"])].signer)
        assert f["test_signer"] not in train_signers
        assert len(f["train_ids"]) + len(f["test_ids"]) == len(idx)


def test_macro_f1_ignores_classes_absent_from_test():
    import numpy as np
    from tempobis.experiment import macro_f1_present

    y_true = np.array([0, 0, 1, 1])            # classes 2 and 3 not in this held-out signer
    assert macro_f1_present(y_true, y_true) == pytest.approx(1.0)
    # predicting an absent class is still an error (lowers recall of class 0)
    assert macro_f1_present(y_true, np.array([0, 3, 1, 1])) < 1.0


def test_run_records_class_coverage_and_skips_hold_speedups(tiny, tmp_path):
    d, idx = tiny
    idx = idx[~((idx.signer == idx.signer.min()) & (idx.label == idx.label.max()))]  # one signer lacks a class
    cfg = _cfg()
    cfg["temporal"]["methods"] = ["uniform"]
    cfg["stress"] = {"factors": [0.5, 1.0, 2.0], "types": ["core", "hold"]}
    seqs, _ = load_sequences(idx, d / "landmarks", cfg)
    res = run(cfg, idx, seqs, tmp_path, verbose=False)
    assert not ((res.perturbation == "hold") & (res.factor > 1)).any()
    assert len(res[res.perturbation == "hold"]) == 3 * 2          # folds x {0.5, 1.0}
    cov = res.groupby("test_signer")["n_test_classes"].first()
    assert cov[idx.signer.min()] == res["n_classes"].iloc[0] - 1
    assert (cov.drop(idx.signer.min()) == res["n_classes"].iloc[0]).all()
    text = write_report(tmp_path, synthetic=True)
    assert "Reduced-coverage folds" in text


def test_run_report_and_resume(tiny):
    d, idx = tiny
    cfg = _cfg(train_speed_aug=True)
    seqs, missing = load_sequences(idx, d / "landmarks", cfg)
    assert not missing
    res = run(cfg, idx, seqs, d / "results", verbose=False)
    n_methods = len(cfg["temporal"]["methods"]) + 1
    expected = 3 * 1 * n_methods * len(cfg["stress"]["types"]) * len(cfg["stress"]["factors"])
    assert len(res) == expected
    assert set(res.method) == {"uniform", "trim_uniform", "motion_aware", "uniform_speedaug"}
    # resume: nothing is re-trained
    res2 = run(cfg, idx, seqs, d / "results", verbose=False)
    assert len(res2) == expected
    text = write_report(d / "results", synthetic=True)
    assert "SYNTHETIC" in text and (d / "results" / "figures" / "robustness_curves.png").exists()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")
def test_gru_trains(tiny, tmp_path):
    d, idx = tiny
    cfg = _cfg("gru", hidden=16, layers=1)
    cfg["temporal"]["methods"] = ["uniform"]
    seqs, _ = load_sequences(idx, d / "landmarks", cfg)
    res = run(cfg, idx, seqs, tmp_path, verbose=False)
    assert len(res) > 0 and res["macro_f1"].between(0, 1).all()
    from tempobis.models import load_model
    m = load_model(next((tmp_path / "models").glob("*.pt")))
    assert m.n_classes == 4
