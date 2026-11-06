import numpy as np
import pandas as pd

from armwm.analysis import early_warning_null, paired_auroc_diff, pre_onset_auroc, residualize
from armwm.metrics import calibrate, online_metrics


def make(n_nom=200, n_def=60, T=60, onset=40, gap=3.0, pre_gap=0.0, seed=0):
    rng = np.random.default_rng(seed)
    scores, rows = {}, []
    for i in range(n_nom):
        scores[f"n{i}"] = rng.normal(0, 1, T)
        rows.append({"pick_id": f"n{i}", "label": "nominal", "onset": -1, "sublabel": ""})
    for i in range(n_def):
        s = rng.normal(0, 1, T) + pre_gap
        s[onset:] += gap
        scores[f"d{i}"] = s
        rows.append({"pick_id": f"d{i}", "label": "open", "onset": onset, "sublabel": "box" if i % 2 else "book"})
    return scores, pd.DataFrame(rows)


def test_paired_diff_zero_for_identical_and_positive_for_better():
    s, idx = make()
    same = paired_auroc_diff(s, s, idx, n_boot=200)
    assert same["diff"] == 0 and same["ci95"] == [0.0, 0.0]
    noise = {k: np.random.default_rng(1).normal(0, 1, len(v)) for k, v in s.items()}
    better = paired_auroc_diff(s, noise, idx, n_boot=200)
    assert better["diff"] > 0.3 and better["ci95"][0] > 0 and better["p"] < 0.01


def test_early_warning_null_matches_chance_for_a_blind_monitor():
    s, idx = make(gap=0.0, n_def=300, seed=2)  # defects look exactly like nominals
    thr = calibrate({k: v for k, v in s.items() if k.startswith("n")}, 0.2)
    r = early_warning_null(s, idx, thr, stride=1, lead_min_s=1.0, fps=30.0)
    assert r["n"] == 300
    assert abs(r["observed_rate"] - r["expected_rate"]) < 0.08
    assert r["p_one_sided"] > 0.01
    # the time-matched null is below the whole-clip false-alarm rate (onset is before clip end)
    assert r["expected_rate"] < r["clip_fa_rate"]


def test_early_warning_null_flags_real_foresight():
    s, idx = make(pre_gap=2.0, seed=3)
    thr = calibrate({k: v for k, v in s.items() if k.startswith("n")}, 0.05)
    r = early_warning_null(s, idx, thr, stride=1)
    assert r["observed_rate"] > r["expected_rate"] + 0.3 and r["p_one_sided"] < 1e-6


def test_pre_onset_auroc_is_chance_without_pre_onset_signal():
    s, idx = make(gap=5.0, seed=4)  # strong post-onset signal must not leak in
    r = pre_onset_auroc(s, idx, stride=1, n_boot=100)
    assert r["n_defect"] == 60 and abs(r["auroc"] - 0.5) < 0.1
    s2, idx2 = make(pre_gap=1.5, seed=4)
    assert pre_onset_auroc(s2, idx2, stride=1, n_boot=100)["auroc"] > 0.8


def test_pre_onset_respects_stride():
    # stride 3: sample i covers source frame 3i; onset 40 -> last pre-onset sample is 13
    s = {"n0": np.zeros(20), "d0": np.r_[np.zeros(14), np.full(6, 9.0)]}
    idx = pd.DataFrame([{"pick_id": "n0", "label": "nominal", "onset": -1},
                        {"pick_id": "d0", "label": "open", "onset": 40}])
    assert pre_onset_auroc(s, idx, stride=3, n_boot=10)["auroc"] == 0.5


def test_residualize_removes_linear_motion():
    rng = np.random.default_rng(5)
    motion = {f"c{i}": rng.normal(0, 1, 50) for i in range(20)}
    score = {k: 2.0 * v + 1.0 + 0.01 * rng.normal(0, 1, 50) for k, v in motion.items()}
    res, fit = residualize(score, motion, list(motion)[:10])
    assert abs(fit["b"] - 2.0) < 0.01 and fit["frame_r2_nominal"] > 0.99
    assert max(np.abs(v).max() for v in res.values()) < 0.1


def test_oracle_detect_rate_sits_at_target():
    s, idx = make(seed=6)
    calib = {k: v for k, v in s.items() if k.startswith("n")}
    r = online_metrics(s, idx, calib, target_fa=0.05)
    assert 0 <= r.oracle_detect_rate <= 1
    assert abs(r.oracle_detect_rate - r.detect_rate) < 0.05  # same nominals: same threshold
