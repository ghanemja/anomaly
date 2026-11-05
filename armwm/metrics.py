"""Evaluation protocol for online defect monitors.

A monitor emits one score per frame, computed causally (frame t may only use
frames <= t). Frames where a monitor has not yet seen enough context are NaN:
no decision, no alarm. Three views of the same scores:

1. **clip**: max score over the clip vs. the clip label (defect / nominal), the
   view comparable to Amazon's clip classifier.
2. **frame**: every frame labelled 1 from the first defect event on (a torn box
   stays torn), pooled AUROC.
3. **online**: one threshold set on *held-out* nominal clips so that a target
   fraction of nominal clips ever alarm. On defect clips we then report the
   first alarm relative to the annotated onset: detection rate, latency, and
   how often the alarm fires *before* onset (early warning, or luck - compare
   with the nominal false-alarm rate).

The official test split has an 11.1% defect rate, close to the operational
rate, so clip precision at a threshold is reported on it as is.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .labels import FPS, frame_labels


def causal_ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average that only looks backwards; NaNs pass through."""
    if alpha >= 1.0:
        return x
    out = np.full_like(x, np.nan, dtype=float)
    acc = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        acc = v if np.isnan(acc) else alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def clip_max(s: np.ndarray) -> float:
    return float(np.nanmax(s)) if np.isfinite(s).any() else -np.inf


def first_alarm(s: np.ndarray, thr: float) -> int:
    hit = np.flatnonzero(np.nan_to_num(s, nan=-np.inf) > thr)
    return int(hit[0]) if hit.size else -1


def _auc(y, s) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def clip_metrics(scores: dict[str, np.ndarray], index: pd.DataFrame, n_boot: int = 0, seed: int = 0) -> dict:
    df = index.set_index("pick_id").loc[list(scores)]
    s = np.array([clip_max(scores[p]) for p in df.index])
    y = (df["label"] != "nominal").to_numpy().astype(int)
    out = {"n_clips": int(len(y)), "n_defect": int(y.sum()), "auroc": _auc(y, s),
           "ap": float(average_precision_score(y, s)) if y.any() else float("nan")}
    for cls in ("open", "deconstruction"):
        m = df["label"].isin(["nominal", cls]).to_numpy()
        out[f"auroc_{cls}"] = _auc(y[m], s[m])
    if n_boot:
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if 0 < y[i].sum() < len(i):
                boots.append(roc_auc_score(y[i], s[i]))
        out["auroc_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def frame_metrics(scores: dict[str, np.ndarray], index: pd.DataFrame, stride: int = 1) -> dict:
    """Pooled frame AUROC. ``scores[p][i]`` is the score of source frame ``i*stride``."""
    onset = index.set_index("pick_id")["onset"]
    ys, ss = [], []
    for p, s in scores.items():
        y = frame_labels(int(onset[p]), len(s) * stride).y[::stride][: len(s)]
        ok = np.isfinite(s)
        ys.append(y[ok])
        ss.append(s[ok])
    y, s = np.concatenate(ys), np.concatenate(ss)
    return {"n_frames": int(len(y)), "frame_auroc": _auc(y, s)}


@dataclass
class OnlineResult:
    target_fa: float
    threshold: float
    fa_test: float  # fraction of test nominal clips that ever alarm
    detect_rate: float  # defect clips with any alarm
    early_rate: float  # defect clips whose first alarm precedes onset
    detect_after_onset_rate: float  # first alarm at/after onset
    within_0_5s: float  # first alarm in [onset, onset+0.5s]
    within_1s: float
    median_latency_s: float  # over clips alarmed at/after onset
    clip_precision: float
    lead_subset_n: int  # defects with >= lead_min_s of pre-onset video
    lead_subset_early_rate: float

    def as_dict(self) -> dict:
        return asdict(self)


def calibrate(calib_scores: dict[str, np.ndarray], target_fa: float) -> float:
    """Threshold at which ``target_fa`` of the calibration nominal clips alarm."""
    m = np.array([clip_max(s) for s in calib_scores.values()])
    return float(np.quantile(m, 1.0 - target_fa))


def online_metrics(
    scores: dict[str, np.ndarray],
    index: pd.DataFrame,
    calib_scores: dict[str, np.ndarray],
    target_fa: float = 0.05,
    stride: int = 1,
    lead_min_s: float = 1.0,
    fps: float = FPS,
) -> OnlineResult:
    thr = calibrate(calib_scores, target_fa)
    df = index.set_index("pick_id").loc[list(scores)]
    nominal = df.index[df["label"] == "nominal"]
    defect = df.index[df["label"] != "nominal"]

    fa = np.mean([first_alarm(scores[p], thr) >= 0 for p in nominal]) if len(nominal) else np.nan
    lat, early, det, w05, w1, lead_n, lead_early = [], 0, 0, 0, 0, 0, 0
    for p in defect:
        s = scores[p]
        onset = min(int(df.at[p, "onset"]), len(s) * stride - 1)
        a = first_alarm(s, thr)
        has_lead = onset / fps >= lead_min_s
        lead_n += has_lead
        if a < 0:
            continue
        det += 1
        t = a * stride
        if t < onset:
            early += 1
            lead_early += has_lead
            continue
        d = (t - onset) / fps
        lat.append(d)
        w05 += d <= 0.5
        w1 += d <= 1.0
    n = max(len(defect), 1)
    tp = det
    fp = fa * len(nominal) if len(nominal) else 0
    return OnlineResult(
        target_fa=target_fa, threshold=thr, fa_test=float(fa),
        detect_rate=det / n, early_rate=early / n, detect_after_onset_rate=len(lat) / n,
        within_0_5s=w05 / n, within_1s=w1 / n,
        median_latency_s=float(np.median(lat)) if lat else float("nan"),
        clip_precision=float(tp / (tp + fp)) if tp + fp else float("nan"),
        lead_subset_n=int(lead_n), lead_subset_early_rate=lead_early / lead_n if lead_n else float("nan"),
    )


def evaluate(
    scores: dict[str, np.ndarray],
    index: pd.DataFrame,
    calib_scores: dict[str, np.ndarray],
    stride: int = 1,
    fa_targets=(0.01, 0.05, 0.10),
    n_boot: int = 1000,
) -> dict:
    return {
        "clip": clip_metrics(scores, index, n_boot=n_boot),
        "frame": frame_metrics(scores, index, stride=stride),
        "online": [online_metrics(scores, index, calib_scores, fa, stride=stride).as_dict() for fa in fa_targets],
    }
