"""Fit and evaluate the CPU baselines on extracted frame features.

Protocol (no test data is used for fitting or thresholds):
  * normal-only scorers are fit on out-of-split nominal clips ("extra");
  * thresholds are calibrated on train-split nominal clips;
  * the supervised reference is fit on 80% of the train split, thresholds on
    the nominal clips of the other 20%;
  * everything is reported on the official test split (11% defects).

    python scripts/run_baselines.py --data data/pilot --labels data/labels
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armwm.features import load_features  # noqa: E402
from armwm.labels import frame_labels, load_index  # noqa: E402
from armwm.metrics import causal_ema, evaluate  # noqa: E402
from armwm.scorers.baselines import FeatureVelocity, KNNDistance, LatentPredictor, Standardizer, SupervisedGRU  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--labels", default="data/labels")
    p.add_argument("--features", default="features_resnet18_s3.npz")
    p.add_argument("--ema", type=float, default=0.5, help="causal smoothing of scores (1 = none)")
    p.add_argument("--out", default="results/pilot_baselines.json")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    feats, stride = load_features(Path(a.data) / a.features)
    idx = load_index(a.labels)
    idx = idx[idx.pick_id.isin(feats)].reset_index(drop=True)
    by = {s: {p: feats[p] for p in idx.pick_id[idx.split == s]} for s in ("extra", "train", "test")}
    lab = idx.set_index("pick_id")
    train_nom = {p: f for p, f in by["train"].items() if lab.at[p, "label"] == "nominal"}
    print({k: len(v) for k, v in by.items()}, "train nominal", len(train_nom),
          "test defects", int((lab.loc[list(by["test"]), "label"] != "nominal").sum()))

    std = Standardizer().fit(by["extra"])
    smooth = lambda s: causal_ema(s, a.ema)  # noqa: E731
    results, scorers, scored = {}, {}, {}

    vel = FeatureVelocity().fit(by["extra"], std)
    knn = KNNDistance(k=5).fit(by["extra"], std)
    scorers["velocity"] = vel.score
    scorers["knn"] = knn.score
    lp = LatentPredictor(context=4, members=5, seed=a.seed).fit(by["extra"], std)
    for name, fn in scorers.items():
        scored[name] = {p: fn(f) for g in ("test", "train") for p, f in by[g].items()}
    lp_all = {p: lp.score_all(f) for g in ("test", "train") for p, f in by[g].items()}
    for kind in ("error", "uncertainty", "surprise"):
        scored[f"latent_pred_{kind}"] = {p: s[kind] for p, s in lp_all.items()}

    for name, sc in scored.items():
        test = {p: smooth(sc[p]) for p in by["test"]}
        calib = {p: smooth(sc[p]) for p in train_nom}
        results[name] = evaluate(test, idx, calib, stride=stride)
        print(name, json.dumps(results[name]["clip"]), flush=True)

    # supervised reference
    rng = np.random.default_rng(a.seed)
    keys = np.array(sorted(by["train"]))
    rng.shuffle(keys)
    cut = int(0.8 * len(keys))
    fit_keys, cal_keys = keys[:cut], keys[cut:]
    tgt = {p: frame_labels(int(lab.at[p, "onset"]), len(by["train"][p]) * stride).y[::stride] for p in keys}
    gru = SupervisedGRU(seed=a.seed).fit({p: by["train"][p] for p in fit_keys}, std, tgt)
    test = {p: smooth(gru.score(f)) for p, f in by["test"].items()}
    calib = {p: smooth(gru.score(by["train"][p])) for p in cal_keys if lab.at[p, "label"] == "nominal"}
    results["supervised_gru"] = evaluate(test, idx, calib, stride=stride)
    print("supervised_gru", json.dumps(results["supervised_gru"]["clip"]), flush=True)

    meta = {"features": a.features, "stride": stride, "ema": a.ema,
            "n": {k: len(v) for k, v in by.items()}, "calib_nominal": len(train_nom)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
