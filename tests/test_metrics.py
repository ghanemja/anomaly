import numpy as np
import pandas as pd
import pytest

from armwm.labels import frame_labels, _event_frames
from armwm.metrics import calibrate, causal_ema, clip_metrics, first_alarm, frame_metrics, online_metrics


def make(n_nom=40, n_def=20, T=60, onset=30, seed=0, gap=3.0):
    rng = np.random.default_rng(seed)
    scores, rows = {}, []
    for i in range(n_nom):
        scores[f"n{i}"] = rng.normal(0, 1, T)
        rows.append({"pick_id": f"n{i}", "label": "nominal", "onset": -1})
    for i in range(n_def):
        s = rng.normal(0, 1, T)
        s[onset:] += gap
        scores[f"d{i}"] = s
        rows.append({"pick_id": f"d{i}", "label": "open" if i % 2 else "deconstruction", "onset": onset})
    calib = {f"c{i}": rng.normal(0, 1, T) for i in range(200)}
    return scores, pd.DataFrame(rows), calib


def test_event_frames_strip_and_null():
    ev = _event_frames([{"frame_idx": 13, "label": "Open "}, {"frame_idx": 28.0, "label": "Deconstruction "},
                        {"frame_idx": None, "label": "nan"}, {"frame_idx": 5, "label": "No Damage"}])
    assert ev == {"open": [13], "deconstruction": [28]}


def test_frame_labels_clamps_onset():
    fl = frame_labels(500, 100)
    assert fl.onset == 99 and fl.y.sum() == 1
    assert frame_labels(-1, 10).y.sum() == 0


def test_causal_ema_is_causal():
    x = np.array([np.nan, 1.0, 1.0, 10.0, 1.0])
    y = causal_ema(x, 0.5)
    assert np.isnan(y[0]) and y[1] == 1.0
    x2 = x.copy(); x2[4] = 1000
    assert np.allclose(causal_ema(x2, 0.5)[:4], y[:4], equal_nan=True)


def test_first_alarm_ignores_nan():
    assert first_alarm(np.array([np.nan, 0.1, 5.0]), 1.0) == 2
    assert first_alarm(np.array([np.nan, np.nan]), -10) == -1


def test_separable_scores_give_high_auc_and_latency():
    scores, idx, calib = make()
    c = clip_metrics(scores, idx, n_boot=100)
    assert c["auroc"] > 0.95 and c["auroc_ci95"][0] <= c["auroc"] <= c["auroc_ci95"][1]
    assert frame_metrics(scores, idx)["frame_auroc"] > 0.9
    r = online_metrics(scores, idx, calib, target_fa=0.05)
    assert r.fa_test < 0.2
    assert r.detect_rate > 0.9 and r.median_latency_s < 0.2


def test_random_scores_are_chance():
    scores, idx, calib = make(n_nom=400, n_def=400, gap=0.0)
    assert abs(clip_metrics(scores, idx)["auroc"] - 0.5) < 0.07


def test_calibrate_hits_target():
    rng = np.random.default_rng(1)
    calib = {i: rng.normal(0, 1, 50) for i in range(2000)}
    thr = calibrate(calib, 0.1)
    assert abs(np.mean([np.nanmax(s) > thr for s in calib.values()]) - 0.1) < 0.01


def test_stride_maps_onset():
    # stride 3: score i covers source frame 3i, so source onset 30 -> index 10
    scores, idx, calib = make(T=60, onset=30)
    s3 = {k: v[::3] for k, v in scores.items()}
    assert frame_metrics(s3, idx, stride=3)["frame_auroc"] > 0.85
    r = online_metrics(s3, idx, {k: v[::3] for k, v in calib.items()}, 0.05, stride=3)
    assert r.detect_rate > 0.9 and r.median_latency_s <= 0.1
