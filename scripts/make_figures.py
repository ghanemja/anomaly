"""Collect the data behind the pilot figures into one JSON file.

Example frames (base64 JPEG), onset timing, per-frame score timelines for a few
clips, and the headline metrics. Feeds the results page.

    python scripts/make_figures.py --data data/pilot --labels data/labels --out results/figures.json
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armwm.features import load_features  # noqa: E402
from armwm.labels import FPS, load_index  # noqa: E402
from armwm.metrics import calibrate, causal_ema  # noqa: E402
from armwm.scorers.baselines import KNNDistance, LatentPredictor, Standardizer  # noqa: E402
from armwm.video import read_frames  # noqa: E402
from armwm.worldmodels.vjepa2 import hold_last  # noqa: E402

TIMELINE = ["001094001655336340247", "001063001655336340247", "010002001653484236499", "011022001654816062991"]
EXAMPLES = ["001094001655336340247", "001063001655336340247", "010002001653484236499", "004033001653324623337"]


def jpeg(frame: np.ndarray, quality: int = 82) -> str:
    img = av.VideoFrame.from_ndarray(frame, format="rgb24").to_image()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--labels", default="data/labels")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/figures.json")
    a = p.parse_args()
    data = Path(a.data)
    idx = load_index(a.labels)
    lab = idx.set_index("pick_id")
    out: dict = {}

    on = idx.loc[idx.onset >= 0, "onset"].to_numpy() / FPS
    edges = np.arange(0, 3.01, 0.1)
    h, _ = np.histogram(np.clip(on, 0, 2.999), edges)
    out["onset"] = {"edges_s": edges.round(2).tolist(), "counts": h.tolist(), "n": int(len(on)),
                    "median_s": float(np.median(on)), "share_le_0_5s": float(np.mean(on <= 0.5)),
                    "share_ge_1s": float(np.mean(on >= 1.0)), "share_over_3s": float(np.mean(on >= 3.0))}

    ex = []
    for pid in EXAMPLES:
        f = read_frames(data / "videos" / f"{pid}.mp4")
        o = min(int(lab.at[pid, "onset"]), len(f) - 1)  # onsets can overshoot the clip
        picks = [("before", max(o - 24, 0)), ("onset", o), ("after", min(o + 24, len(f) - 1)), ("end", len(f) - 1)]
        ex.append({"pick_id": pid, "label": lab.at[pid, "label"], "item": lab.at[pid, "sublabel"], "onset_s": o / FPS,
                   "frames": [{"tag": t, "t_s": i / FPS, "src": jpeg(f[i])} for t, i in picks]})
    out["examples"] = ex

    feats, stride = load_features(data / "features_resnet18_s3.npz")
    extra = {k: v for k, v in feats.items() if k in lab.index and lab.at[k, "split"] == "extra"}
    calib_ids = [k for k in feats if k in lab.index and lab.at[k, "split"] == "train" and lab.at[k, "label"] == "nominal"]
    std = Standardizer().fit(extra)
    knn = KNNDistance(k=5).fit(extra, std)
    lp = LatentPredictor(context=4, members=5, seed=0).fit(extra, std)
    sm = lambda s: causal_ema(s, 0.5)  # noqa: E731
    mons = {
        "knn": lambda f: sm(knn.score(f)),
        "latent_pred_uncertainty": lambda f: sm(lp.score_all(f)["uncertainty"]),
    }
    thr = {m: calibrate({k: fn(feats[k]) for k in calib_ids}, 0.05) for m, fn in mons.items()}
    vj = {}
    vdir = data / "vjepa2_scores"
    vcal = [p.stem for p in vdir.glob("*.npz") if lab.at[p.stem, "split"] == "train"][:15]
    if vcal:
        load = lambda k: causal_ema(hold_last(np.load(vdir / f"{k}.npz")["max"]), 0.5)  # noqa: E731
        thr["vjepa2_max"] = calibrate({k: load(k) for k in vcal}, 0.05)
        vj = {k: load(k) for k in TIMELINE if (vdir / f"{k}.npz").exists()}
    tl = []
    for pid in TIMELINE:
        ser = {m: fn(feats[pid]) for m, fn in mons.items()}
        t = (np.arange(len(feats[pid])) * stride / FPS)
        row = {"pick_id": pid, "label": lab.at[pid, "label"], "item": lab.at[pid, "sublabel"],
               "onset_s": float(lab.at[pid, "onset"] / FPS) if lab.at[pid, "onset"] >= 0 else None,
               "series": {m: {"t": t.round(3).tolist(), "v": [None if not np.isfinite(x) else round(float(x), 5) for x in s]} for m, s in ser.items()}}
        if pid in vj:
            s = vj[pid]
            keep = np.flatnonzero(np.isfinite(s))[::2]
            row["series"]["vjepa2_max"] = {"t": (keep / FPS).round(3).tolist(), "v": [round(float(s[i]), 5) for i in keep]}
        tl.append(row)
    out["timelines"] = tl
    out["thresholds_5pct"] = thr

    res = {}
    for fn in ("pilot_baselines.json", "pilot_vjepa2_cpu.json"):
        fp = Path(a.results) / fn
        if fp.exists():
            for k, v in json.loads(fp.read_text())["results"].items():
                o5 = next(o for o in v["online"] if o["target_fa"] == 0.05)
                res[k] = {"auroc": v["clip"]["auroc"], "ci": v["clip"].get("auroc_ci95"), "n": v["clip"]["n_clips"],
                          "auroc_open": v["clip"]["auroc_open"], "auroc_decon": v["clip"]["auroc_deconstruction"],
                          "detect_5": o5["detect_rate"], "fa_test_5": o5["fa_test"], "early_5": o5["early_rate"]}
    out["results"] = res
    Path(a.out).write_text(json.dumps(out))
    print("wrote", a.out, round(Path(a.out).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
