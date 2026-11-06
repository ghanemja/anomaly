"""Controls and significance tests for the online-monitor results.

Each function answers one objection a reviewer would raise about a headline
claim:

* ``paired_auroc_diff``: are two monitors really tied, or is one better?
  Overlapping per-monitor intervals are not a test; this bootstraps the
  *difference* on the same resampled clips.
* ``early_warning_null``: is "alarm before onset" foresight or chance? The
  clip-level false-alarm rate covers a whole ~4.5 s clip, but a defect clip only
  has ``onset`` frames in which to alarm early. The right null is the share of
  nominal clips that alarm before that same frame.
* ``pre_onset_auroc``: does a monitor separate defect clips from nominal clips
  *before any damage is visible*? If so it is reacting to something other than
  the defect (item type, grasp, scene) or it has real foresight. Either way the
  post-onset numbers are not purely defect detection. Nominal clips carry no
  item labels in ARMBench, so this label-free test stands in for a per-item
  breakdown.
* ``residualize``: how much of a monitor is plain motion energy? Regress its
  per-frame score on the velocity score over nominal frames and keep the
  residual as a new monitor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .metrics import clip_max, first_alarm


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with tie correction (fast enough to bootstrap)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def clip_arrays(scores: dict[str, np.ndarray], index: pd.DataFrame, ids=None):
    ids = list(scores) if ids is None else list(ids)
    lab = index.set_index("pick_id").loc[ids, "label"]
    y = (lab != "nominal").to_numpy().astype(int)
    s = np.array([clip_max(scores[p]) for p in ids])
    return ids, y, s


def paired_auroc_diff(a: dict[str, np.ndarray], b: dict[str, np.ndarray], index: pd.DataFrame,
                      n_boot: int = 2000, seed: int = 0) -> dict:
    """AUROC(a) - AUROC(b) on the same clips, with a paired bootstrap CI and p-value.

    The p-value is two-sided: twice the smaller tail share of bootstrap
    differences on the far side of zero.
    """
    ids = sorted(set(a) & set(b))
    _, y, sa = clip_arrays(a, index, ids)
    _, _, sb = clip_arrays(b, index, ids)
    d0 = _auc(y, sa) - _auc(y, sb)
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            ds.append(_auc(y[i], sa[i]) - _auc(y[i], sb[i]))
    ds = np.array(ds)
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return {"diff": float(d0), "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
            "p": float(min(p, 1.0)), "n_clips": len(ids)}


def _poisson_binomial_sf(p: np.ndarray, k: int) -> float:
    """P(sum of independent Bernoulli(p_i) >= k), exact by dynamic programming."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for q in p:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= 1 - q
    return float(dist[k:].sum())


def early_warning_null(scores: dict[str, np.ndarray], index: pd.DataFrame, threshold: float,
                       stride: int = 1, lead_min_s: float = 1.0, fps: float = 30.0) -> dict:
    """Observed vs. chance rate of alarms before onset, matched in time.

    For each defect clip with at least ``lead_min_s`` of pre-onset video, the
    chance of an early alarm is the share of nominal clips in ``scores`` that
    alarm before that clip's onset frame. Summing those chances gives the
    expected number of early alarms if the monitor knew nothing; the observed
    count is tested against it (exact one-sided Poisson-binomial tail).
    """
    df = index.set_index("pick_id").loc[list(scores)]
    nominal = [p for p in df.index if df.at[p, "label"] == "nominal"]
    nom_first = np.array([first_alarm(scores[p], threshold) for p in nominal])
    nom_first = np.where(nom_first < 0, np.iinfo(np.int64).max, nom_first * stride)
    chance, observed = [], 0
    for p in df.index[df["label"] != "nominal"]:
        onset = int(df.at[p, "onset"])
        if onset / fps < lead_min_s:
            continue
        chance.append(float(np.mean(nom_first < onset)))
        a = first_alarm(scores[p], threshold)
        observed += int(0 <= a * stride < onset)
    chance = np.array(chance)
    n = len(chance)
    return {"n": n, "observed": observed, "observed_rate": observed / n if n else float("nan"),
            "expected": float(chance.sum()), "expected_rate": float(chance.mean()) if n else float("nan"),
            "clip_fa_rate": float(np.mean(nom_first < np.iinfo(np.int64).max)) if len(nominal) else float("nan"),
            "p_one_sided": _poisson_binomial_sf(chance, observed) if n else float("nan")}


def _prefix_max(s: np.ndarray) -> np.ndarray:
    return np.fmax.accumulate(np.nan_to_num(s, nan=-np.inf))


def pre_onset_auroc(scores: dict[str, np.ndarray], index: pd.DataFrame, stride: int = 1,
                    lead_min_s: float = 1.0, fps: float = 30.0, n_boot: int = 1000, seed: int = 0) -> dict:
    """AUROC of defect vs. nominal clips using only frames before the defect onset.

    Each defect clip (with at least ``lead_min_s`` before onset) is scored by its
    max over frames before onset, and compared with every nominal clip's max
    over the *same* number of frames, so clip length and warm-up cannot bias it.
    0.5 means the monitor cannot tell, before the damage, which pick will fail.
    """
    df = index.set_index("pick_id").loc[list(scores)]
    nom = [_prefix_max(scores[p]) for p in df.index[df["label"] == "nominal"]]
    L = max(len(x) for x in nom)
    N = np.full((len(nom), L), -np.inf)
    for i, x in enumerate(nom):
        N[i, : len(x)] = x
        N[i, len(x):] = x[-1]
    D, K = [], []  # pre-onset max of each defect; index of its last pre-onset sample
    for p in df.index[df["label"] != "nominal"]:
        onset = int(df.at[p, "onset"])
        if onset / fps < lead_min_s:
            continue
        k = (onset - 1) // stride  # last scored sample strictly before onset
        pm = _prefix_max(scores[p])
        k = min(k, len(pm) - 1)
        if not np.isfinite(pm[k]):
            continue
        D.append(pm[k])
        K.append(k)
    D, K = np.array(D), np.array(K)
    if not len(D):
        return {"n_defect": 0, "auroc": float("nan")}
    ref = N[:, K]  # (n_nominal, n_defect): nominal max over the same window
    win = (D[None, :] > ref) + 0.5 * (D[None, :] == ref)
    auc = float(win.mean())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, win.shape[0], win.shape[0])
        j = rng.integers(0, win.shape[1], win.shape[1])
        boots.append(win[np.ix_(i, j)].mean())
    return {"n_defect": int(len(D)), "n_nominal": int(win.shape[0]), "auroc": auc,
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def residualize(scores: dict[str, np.ndarray], motion: dict[str, np.ndarray], fit_ids) -> tuple[dict, dict]:
    """Remove the part of a per-frame score that a linear function of motion explains.

    Fits ``score ~ a + b * motion`` on the frames of ``fit_ids`` (nominal,
    non-test clips) and returns the residual scores for every clip plus the fit
    (coefficients and frame-level R^2).
    """
    x = np.concatenate([motion[p] for p in fit_ids])
    y = np.concatenate([scores[p] for p in fit_ids])
    ok = np.isfinite(x) & np.isfinite(y)
    b, a = np.polyfit(x[ok], y[ok], 1)
    r2 = float(np.corrcoef(x[ok], y[ok])[0, 1] ** 2)
    out = {p: scores[p] - (a + b * motion[p]) for p in scores}
    return out, {"a": float(a), "b": float(b), "frame_r2_nominal": r2}
