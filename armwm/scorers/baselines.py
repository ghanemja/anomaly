"""Causal per-frame anomaly scorers over frame embeddings.

Every scorer maps a clip's ``(T, D)`` embeddings to ``(T,)`` scores where the
score at t depends only on frames <= t. Normal-only scorers are fit on nominal
clips (the out-of-split pool); ``SupervisedGRU`` is a labelled reference that
plays the role of Amazon's clip classifier.
"""
from __future__ import annotations

import numpy as np


class Standardizer:
    def fit(self, feats: dict[str, np.ndarray]) -> "Standardizer":
        x = np.concatenate([f.astype(np.float32) for f in feats.values()])
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-4
        return self

    def __call__(self, f: np.ndarray) -> np.ndarray:
        return (f.astype(np.float32) - self.mu) / self.sd


class FeatureVelocity:
    """Size of the frame-to-frame change in embedding space (a motion-energy baseline)."""

    name = "velocity"

    def fit(self, feats, std: Standardizer):
        self.std = std
        return self

    def score(self, f: np.ndarray) -> np.ndarray:
        z = self.std(f)
        s = np.full(len(z), np.nan)
        s[1:] = np.linalg.norm(np.diff(z, axis=0), axis=1)
        return s


class KNNDistance:
    """Mean cosine distance to the k nearest nominal frames (global PatchCore-style)."""

    name = "knn"

    def __init__(self, k: int = 5, bank_size: int = 60_000, seed: int = 0):
        self.k, self.bank_size, self.seed = k, bank_size, seed

    @staticmethod
    def _unit(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    def fit(self, feats, std: Standardizer):
        self.std = std
        bank = np.concatenate([std(f) for f in feats.values()])
        if len(bank) > self.bank_size:
            bank = bank[np.random.default_rng(self.seed).choice(len(bank), self.bank_size, replace=False)]
        self.bank = self._unit(bank).astype(np.float32)
        return self

    def score(self, f: np.ndarray) -> np.ndarray:
        q = self._unit(self.std(f)).astype(np.float32)
        sim = q @ self.bank.T
        top = -np.partition(-sim, self.k - 1, axis=1)[:, : self.k]
        return 1.0 - top.mean(1)


class LatentPredictor:
    """A small action-free world model in embedding space.

    An ensemble of MLPs predicts the next frame's embedding from the previous
    ``context`` embeddings, trained on nominal clips only. Three scores come
    out of the same models, matching the open question in the literature
    (prediction error vs. model uncertainty):

    * ``error``: mean squared error of the ensemble-mean prediction;
    * ``uncertainty``: ensemble disagreement (variance across members);
    * ``surprise``: error divided by uncertainty.

    The first ``context`` frames of every clip are NaN (no decision).
    """

    name = "latent_pred"

    def __init__(self, context: int = 4, hidden: int = 512, members: int = 5, epochs: int = 15,
                 lr: float = 1e-3, pca_dim: int = 128, seed: int = 0):
        self.context, self.hidden, self.members = context, hidden, members
        self.epochs, self.lr, self.pca_dim, self.seed = epochs, lr, pca_dim, seed

    def _windows(self, z: np.ndarray):
        c = self.context
        if len(z) <= c:
            return None, None
        idx = np.arange(c)[None, :] + np.arange(len(z) - c)[:, None]
        return z[idx].reshape(len(idx), -1), z[c:]

    def fit(self, feats, std: Standardizer):
        import torch

        self.std = std
        x = np.concatenate([std(f) for f in feats.values()])
        # PCA keeps the predictor small and the error dominated by real structure
        self.mu = x.mean(0)
        _, _, vt = np.linalg.svd(x[:: max(1, len(x) // 50_000)] - self.mu, full_matrices=False)
        self.proj = vt[: self.pca_dim].T.astype(np.float32)
        X, Y = zip(*(self._windows(self._pca(f)) for f in feats.values()))
        X = np.concatenate([a for a in X if a is not None])
        Y = np.concatenate([a for a in Y if a is not None])
        Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
        d_in, d = X.shape[1], Y.shape[1]
        self.nets = []
        for m in range(self.members):
            g = torch.Generator().manual_seed(self.seed + m)
            torch.manual_seed(self.seed + m)
            net = torch.nn.Sequential(
                torch.nn.Linear(d_in, self.hidden), torch.nn.GELU(),
                torch.nn.Linear(self.hidden, self.hidden), torch.nn.GELU(),
                torch.nn.Linear(self.hidden, d),
            )
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
            boot = torch.randint(0, len(Xt), (len(Xt),), generator=g)  # bootstrap per member
            for _ in range(self.epochs):
                perm = boot[torch.randperm(len(boot), generator=g)]
                for i in range(0, len(perm), 512):
                    b = perm[i : i + 512]
                    loss = torch.nn.functional.mse_loss(net(Xt[b]), Yt[b])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            self.nets.append(net.eval())
        return self

    def _pca(self, f):
        return ((self.std(f) - self.mu) @ self.proj).astype(np.float32)

    def score_all(self, f: np.ndarray) -> dict[str, np.ndarray]:
        import torch

        z = self._pca(f)
        T = len(z)
        out = {k: np.full(T, np.nan) for k in ("error", "uncertainty", "surprise")}
        X, Y = self._windows(z)
        if X is None:
            return out
        with torch.inference_mode():
            P = np.stack([n(torch.from_numpy(X)).numpy() for n in self.nets])
        mean = P.mean(0)
        err = ((mean - Y) ** 2).mean(1)
        unc = P.var(0).mean(1)
        c = self.context
        out["error"][c:] = err
        out["uncertainty"][c:] = unc
        out["surprise"][c:] = err / (unc + 1e-6)
        return out


class SupervisedGRU:
    """Labelled reference: a causal GRU emitting P(defect has happened) per frame.

    Trained with per-frame targets (1 from onset on), so it is an online
    classifier, not a normal-only monitor. Use it to bound what labels buy.
    """

    name = "supervised_gru"

    def __init__(self, hidden: int = 128, epochs: int = 20, lr: float = 1e-3, pca_dim: int = 128, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.pca_dim, self.seed = hidden, epochs, lr, pca_dim, seed

    def fit(self, feats, std: Standardizer, frame_targets: dict[str, np.ndarray]):
        import torch

        self.std = std
        x = np.concatenate([std(f) for f in feats.values()])
        self.mu = x.mean(0)
        _, _, vt = np.linalg.svd(x[:: max(1, len(x) // 50_000)] - self.mu, full_matrices=False)
        self.proj = vt[: self.pca_dim].T.astype(np.float32)
        torch.manual_seed(self.seed)
        self.gru = torch.nn.GRU(self.pca_dim, self.hidden, batch_first=True)
        self.head = torch.nn.Linear(self.hidden, 1)
        params = list(self.gru.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        keys = list(feats)
        pos = np.mean([frame_targets[k].mean() for k in keys])
        pw = torch.tensor((1 - pos) / max(pos, 1e-3))
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            for i in range(0, len(keys), 32):
                b = [keys[j] for j in rng.permutation(len(keys))[i : i + 32]]
                xs = [torch.from_numpy(self._pca(feats[k])) for k in b]
                ys = [torch.from_numpy(frame_targets[k][: len(feats[k])].astype(np.float32)) for k in b]
                X = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
                Y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True)
                M = torch.nn.utils.rnn.pad_sequence([torch.ones(len(y)) for y in ys], batch_first=True)
                logit = self.head(self.gru(X)[0]).squeeze(-1)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, Y, pos_weight=pw, reduction="none") * M).sum() / M.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def _pca(self, f):
        return ((self.std(f) - self.mu) @ self.proj).astype(np.float32)

    def score(self, f: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            h = self.gru(torch.from_numpy(self._pca(f))[None])[0]
            return torch.sigmoid(self.head(h)).squeeze().numpy().reshape(-1).astype(float)
