import numpy as np
import pytest

from armwm.scorers.baselines import FeatureVelocity, KNNDistance, LatentPredictor, Standardizer, SupervisedGRU

torch = pytest.importorskip("torch")


def clips(n, T=40, D=32, seed=0, anomaly_at=None):
    """Smooth trajectories along a fixed direction; anomalies jump off-manifold."""
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(99).normal(size=(2, D))
    out = {}
    for i in range(n):
        t = np.linspace(0, 1, T)[:, None]
        f = t * base[0] + (1 - t) * base[1] + rng.normal(0, 0.05, (T, D))
        if anomaly_at is not None:
            f[anomaly_at:] += rng.normal(0, 1.5, D)
        out[f"c{seed}_{i}"] = f.astype(np.float16)
    return out


def test_all_scorers_shapes_and_signal():
    train = clips(30)
    std = Standardizer().fit(train)
    nom, ano = clips(5, seed=1), clips(5, seed=2, anomaly_at=20)
    for sc in [FeatureVelocity().fit(train, std), KNNDistance(k=3).fit(train, std)]:
        s_n = np.nanmean([np.nanmax(sc.score(f)) for f in nom.values()])
        s_a = np.nanmean([np.nanmax(sc.score(f)) for f in ano.values()])
        assert s_a > s_n, sc.name
    lp = LatentPredictor(context=3, hidden=64, members=2, epochs=3, pca_dim=8).fit(train, std)
    out = lp.score_all(next(iter(ano.values())))
    assert set(out) == {"error", "uncertainty", "surprise"}
    assert np.isnan(out["error"][:3]).all() and np.isfinite(out["error"][3:]).all()
    e_n = np.mean([np.nanmax(lp.score_all(f)["error"]) for f in nom.values()])
    e_a = np.mean([np.nanmax(lp.score_all(f)["error"]) for f in ano.values()])
    assert e_a > e_n


def test_supervised_gru_is_causal():
    train = {**clips(20), **clips(20, seed=3, anomaly_at=20)}
    tgt = {k: (np.arange(40) >= 20).astype(np.int8) if k.startswith("c3_") else np.zeros(40, np.int8) for k in train}
    g = SupervisedGRU(hidden=16, epochs=3, pca_dim=8).fit(train, Standardizer().fit(train), tgt)
    f = next(iter(clips(1, seed=4).values()))
    a = g.score(f)
    f2 = f.copy(); f2[30:] += 5
    assert np.allclose(g.score(f2)[:30], a[:30], atol=1e-6)
