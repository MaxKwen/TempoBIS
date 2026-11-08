"""Classifiers with one interface: fit(X, y, X_val, y_val) -> self, predict_proba(X).

X has shape (n, T, D): fixed-length sequences from temporal.normalize.

  GRUClassifier - bidirectional GRU + mean/max pooling (PyTorch). Main model.
  MLPClassifier - flatten + scikit-learn MLP. "Simple first" baseline; also lets the
                  whole pipeline run where PyTorch is not installed.
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


class _Standardizer:
    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(0)
        self.std = flat.std(0) + 1e-6
        return self

    def __call__(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)


# ---------------------------------------------------------------- scikit-learn
class MLPClassifier:
    def __init__(self, n_classes: int, hidden: int = 256, seed: int = 0, **_):
        self.n_classes, self.hidden, self.seed = n_classes, hidden, seed

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.neural_network import MLPClassifier as SkMLP

        self.std = _Standardizer().fit(X)
        self.clf = SkMLP(hidden_layer_sizes=(self.hidden,), alpha=1e-3, max_iter=400,
                         early_stopping=X_val is None, random_state=self.seed)
        self.clf.fit(self.std(X).reshape(len(X), -1), y)
        self.history = []
        return self

    def predict_proba(self, X):
        p = self.clf.predict_proba(self.std(X).reshape(len(X), -1))
        out = np.zeros((len(X), self.n_classes))
        out[:, self.clf.classes_] = p
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)


# ---------------------------------------------------------------- PyTorch
def _build_net(in_dim, n_classes, hidden, layers, dropout):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                     nn.Dropout(dropout))
            self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                              bidirectional=True, dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_classes))

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            pooled = torch.cat([h.mean(1), h.amax(1)], dim=1)
            return self.head(pooled)

    return Net()


class GRUClassifier:
    def __init__(self, n_classes: int, hidden=128, layers=2, dropout=0.3, epochs=80, batch_size=32,
                 lr=1e-3, weight_decay=1e-4, patience=15, seed=0, device=None, verbose=False, **_):
        self.kw = dict(hidden=hidden, layers=layers, dropout=dropout)
        self.n_classes, self.epochs, self.bs = n_classes, epochs, batch_size
        self.lr, self.wd, self.patience, self.seed = lr, weight_decay, patience, seed
        self.device, self.verbose = device, verbose

    def _dev(self):
        import torch

        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y, X_val=None, y_val=None):
        import torch
        import torch.nn.functional as Fn

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = self._dev()
        self.std = _Standardizer().fit(X)
        Xt = torch.tensor(self.std(X))
        yt = torch.tensor(y, dtype=torch.long)
        self.net = _build_net(X.shape[-1], self.n_classes, **self.kw).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        best, best_state, bad = -1.0, None, 0
        self.history = []
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.tensor(rng.permutation(len(Xt)))
            total = 0.0
            for i in range(0, len(perm), self.bs):
                idx = perm[i:i + self.bs]
                xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
                # light feature-space augmentation (same for every temporal method)
                xb = xb * (1 + 0.05 * torch.randn(xb.size(0), 1, 1, device=dev)) \
                    + 0.02 * torch.randn_like(xb)
                loss = Fn.cross_entropy(self.net(xb), yb, label_smoothing=0.1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
            sched.step()
            rec = {"epoch": ep, "loss": total / len(Xt)}
            if X_val is not None:
                f1 = f1_score(y_val, self.predict_proba(X_val).argmax(1), average="macro")
                rec["val_macro_f1"] = f1
                if f1 > best + 1e-4:
                    best, best_state, bad = f1, copy.deepcopy(self.net.state_dict()), 0
                else:
                    bad += 1
            self.history.append(rec)
            if self.verbose and ep % 10 == 0:
                print(rec)
            if X_val is not None and bad >= self.patience:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        import torch

        self.net.eval()
        out = []
        with torch.no_grad():
            Xt = torch.tensor(self.std(X))
            for i in range(0, len(Xt), 256):
                out.append(torch.softmax(self.net(Xt[i:i + 256].to(self._dev())), 1).cpu().numpy())
        return np.concatenate(out)

    def save(self, path):
        import torch

        torch.save({"state": self.net.state_dict(), "kw": self.kw, "n_classes": self.n_classes,
                    "mean": self.std.mean, "std": self.std.std, "in_dim": len(self.std.mean)}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        import torch

        ck = torch.load(path, map_location=device, weights_only=False)
        m = cls(ck["n_classes"], device=device, **ck["kw"])
        m.std = _Standardizer()
        m.std.mean, m.std.std = ck["mean"], ck["std"]
        m.net = _build_net(ck["in_dim"], ck["n_classes"], **ck["kw"]).to(device)
        m.net.load_state_dict(ck["state"])
        return m


def make_model(cfg_model: dict, n_classes: int, seed: int):
    kind = cfg_model["type"]
    kw = {k: v for k, v in cfg_model.items() if k not in ("type", "train_speed_aug", "train_full_aug")}
    if kind == "gru":
        return GRUClassifier(n_classes, seed=seed, **kw)
    if kind == "mlp":
        return MLPClassifier(n_classes, seed=seed)
    raise ValueError(f"unknown model type: {kind}")


def load_model(path: str | Path):
    path = Path(path)
    if path.suffix == ".pt":
        return GRUClassifier.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)
