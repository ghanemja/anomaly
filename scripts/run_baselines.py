"""Fit and evaluate the CPU baselines on extracted frame features.

Protocol (no test data is used for fitting or thresholds):
  * normal-only scorers are fit on out-of-split nominal clips ("extra");
  * thresholds are calibrated on train-split nominal clips;
  * the supervised reference is cross-fitted on the train split (5 folds), so
    every train nominal clip gets an out-of-fold score and the GRU is
    calibrated on the same ~950 nominal clips as every other monitor. Each test
    clip is scored by one fold model (assigned round-robin), so test and
    calibration scores come from the same kind of model;
  * "<monitor>_resid" variants remove the part of the per-frame score that a
    linear function of the velocity (motion-energy) score explains, fitted on
    train nominal frames;
  * everything is reported on the official test split (11% defects), plus the
    controls in ``armwm.analysis`` (paired tests, time-matched early-warning
    null, pre-onset AUROC).

All settings below are fixed defaults (see docs/RESEARCH_PLAN.md, "Fixed
settings"); none was tuned on test.

    python scripts/run_baselines.py --data data/pilot --labels data/labels
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armwm.analysis import early_warning_null, paired_auroc_diff, pre_onset_auroc, residualize  # noqa: E402
from armwm.features import load_features  # noqa: E402
from armwm.labels import frame_labels, load_index  # noqa: E402
from armwm.metrics import calibrate, causal_ema, clip_max, evaluate  # noqa: E402
from armwm.scorers.baselines import FeatureVelocity, KNNDistance, LatentPredictor, Standardizer, SupervisedGRU  # noqa: E402

FOLDS = 5


def save_scores(path: Path, scored: dict[str, dict[str, np.ndarray]]) -> None:
    """Per-frame smoothed scores, one array per (monitor, clip), as float32.

    float16 cannot resolve probabilities near 1, which ties the supervised
    scores and shifts their AUROC; float32 reproduces the JSON exactly.
    """
    np.savez_compressed(path, **{f"{m}/{p}": s.astype(np.float32) for m, d in scored.items() for p, s in d.items()})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--labels", default="data/labels")
    p.add_argument("--features", default="features_resnet18_s3.npz")
    p.add_argument("--ema", type=float, default=0.5, help="causal smoothing of scores (1 = none)")
    p.add_argument("--out", default="results/pilot_baselines.json")
    p.add_argument("--scores-out", default="results/pilot_scores.npz", help="per-clip scores, read by make_site_data.py")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    feats, stride = load_features(Path(a.data) / a.features)
    idx = load_index(a.labels)
    idx = idx[idx.pick_id.isin(feats)].reset_index(drop=True)
    by = {s: {p: feats[p] for p in idx.pick_id[idx.split == s]} for s in ("extra", "train", "test")}
    lab = idx.set_index("pick_id")
    train_nom = [p for p in by["train"] if lab.at[p, "label"] == "nominal"]
    print({k: len(v) for k, v in by.items()}, "train nominal", len(train_nom),
          "test defects", int((lab.loc[list(by["test"]), "label"] != "nominal").sum()), flush=True)

    std = Standardizer().fit(by["extra"])
    raw: dict[str, dict[str, np.ndarray]] = {}  # unsmoothed per-frame scores on test + train clips
    clips = {p: f for g in ("test", "train") for p, f in by[g].items()}

    vel = FeatureVelocity().fit(by["extra"], std)
    knn = KNNDistance(k=5).fit(by["extra"], std)
    raw["velocity"] = {p: vel.score(f) for p, f in clips.items()}
    raw["knn"] = {p: knn.score(f) for p, f in clips.items()}
    lp = LatentPredictor(context=4, members=5, seed=a.seed).fit(by["extra"], std)
    lp_all = {p: lp.score_all(f) for p, f in clips.items()}
    for kind in ("error", "uncertainty", "surprise"):
        raw[f"latent_pred_{kind}"] = {p: s[kind] for p, s in lp_all.items()}

    # supervised reference, cross-fitted on the train split
    rng = np.random.default_rng(a.seed)
    keys = np.array(sorted(by["train"]))
    rng.shuffle(keys)
    fold_of = {k: i % FOLDS for i, k in enumerate(keys)}
    tgt = {k: frame_labels(int(lab.at[k, "onset"]), len(by["train"][k]) * stride).y[::stride] for k in keys}
    gru_scores = {}
    test_keys = sorted(by["test"])
    for fold in range(FOLDS):
        fit = {k: by["train"][k] for k in keys if fold_of[k] != fold}
        gru = SupervisedGRU(seed=a.seed + fold).fit(fit, std, tgt)
        gru_scores.update({k: gru.score(by["train"][k]) for k in keys if fold_of[k] == fold})
        gru_scores.update({k: gru.score(by["test"][k]) for i, k in enumerate(test_keys) if i % FOLDS == fold})
        print("gru fold", fold, "done", flush=True)
    raw["supervised_gru"] = gru_scores

    # motion-controlled variants
    motion_fits = {}
    for name in ("knn", "latent_pred_error", "latent_pred_uncertainty", "supervised_gru"):
        raw[f"{name}_resid"], motion_fits[name] = residualize(raw[name], raw["velocity"], train_nom)

    smooth = lambda s: causal_ema(s, a.ema)  # noqa: E731
    scored = {m: {p: smooth(s) for p, s in d.items()} for m, d in raw.items()}
    results = {}
    for name, sc in scored.items():
        test = {p: sc[p] for p in by["test"]}
        calib = {p: sc[p] for p in train_nom}
        results[name] = evaluate(test, idx, calib, stride=stride)
        thr5 = calibrate(calib, 0.05)
        results[name]["early_warning_null_5"] = early_warning_null(test, idx, thr5, stride=stride)
        results[name]["pre_onset"] = pre_onset_auroc(test, idx, stride=stride)
        print(name, json.dumps(results[name]["clip"]), flush=True)

    # controls across monitors
    T = {m: {p: scored[m][p] for p in by["test"]} for m in scored}
    pairs = [("supervised_gru", "latent_pred_error"), ("supervised_gru", "latent_pred_uncertainty"),
             ("supervised_gru", "knn"), ("latent_pred_error", "latent_pred_uncertainty"),
             ("latent_pred_error", "knn"), ("latent_pred_error", "velocity"),
             ("latent_pred_error", "latent_pred_error_resid"), ("supervised_gru", "supervised_gru_resid")]
    paired = {f"{x} - {y}": paired_auroc_diff(T[x], T[y], idx, seed=a.seed) for x, y in pairs}
    vmax = np.array([clip_max(T["velocity"][p]) for p in test_keys])
    motion_corr = {m: float(np.corrcoef(vmax, [clip_max(T[m][p]) for p in test_keys])[0, 1])
                   for m in T if m != "velocity"}
    for k, v in paired.items():
        print(k, v, flush=True)

    meta = {"features": a.features, "stride": stride, "ema": a.ema, "seed": a.seed, "gru_folds": FOLDS,
            "n": {k: len(v) for k, v in by.items()}, "calib_nominal": len(train_nom)}
    controls = {"paired_auroc": paired, "clip_max_corr_with_velocity": motion_corr, "motion_fit": motion_fits}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "results": results, "controls": controls}, indent=1))
    save_scores(Path(a.scores_out), {m: {p: d[p] for p in list(by["test"]) + train_nom} for m, d in scored.items()})
    print("wrote", a.out, a.scores_out)


if __name__ == "__main__":
    main()
