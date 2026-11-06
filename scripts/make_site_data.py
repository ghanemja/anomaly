"""Export everything the GitHub Pages site needs into docs/.

Writes docs/data.json plus small JPEGs under docs/img/:
  * player clips: every 3rd frame (10 fps) of a few test clips, with each
    monitor's per-frame score;
  * a defect wall: one thumbnail and a four-frame strip for 40 random test defects;
  * per-clip max scores for every test clip and calibration clip, so the page
    can move the alarm threshold live;
  * onset timing and the headline results.

    python scripts/make_site_data.py --data data/pilot --labels data/labels

Frames are from ARMBench (CC BY 4.0, Amazon Robotics), credited on the page.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armwm.features import load_features  # noqa: E402
from armwm.labels import FPS, frame_labels, load_index  # noqa: E402
from armwm.metrics import causal_ema  # noqa: E402
from armwm.scorers.baselines import (FeatureVelocity, KNNDistance, LatentPredictor,  # noqa: E402
                                     Standardizer, SupervisedGRU)
from armwm.video import read_frames  # noqa: E402
from armwm.worldmodels.vjepa2 import hold_last  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MISSED = "001063001655336340247"  # box opens, every monitor stays quiet
NORMAL = "011022001654816062991"
WALL_N = 40


def save_jpg(frame: np.ndarray, path: Path, quality: int = 74) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    av.VideoFrame.from_ndarray(frame, format="rgb24").to_image().save(path, format="JPEG", quality=quality)


def r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--labels", default="data/labels")
    p.add_argument("--out", default=ROOT / "docs")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    data, out = Path(a.data), Path(a.out)
    img = out / "img"
    shutil.rmtree(img, ignore_errors=True)
    idx = load_index(a.labels)
    lab = idx.set_index("pick_id")

    feats, stride = load_features(data / "features_resnet18_s3.npz")
    by = {s: {k: v for k, v in feats.items() if k in lab.index and lab.at[k, "split"] == s} for s in ("extra", "train", "test")}
    train_nom = [k for k in by["train"] if lab.at[k, "label"] == "nominal"]

    # monitors, fitted exactly as in run_baselines.py
    std = Standardizer().fit(by["extra"])
    vel = FeatureVelocity().fit(by["extra"], std)
    knn = KNNDistance(k=5).fit(by["extra"], std)
    lp = LatentPredictor(context=4, members=5, seed=a.seed).fit(by["extra"], std)
    rng = np.random.default_rng(a.seed)
    keys = np.array(sorted(by["train"]))
    rng.shuffle(keys)
    cut = int(0.8 * len(keys))
    tgt = {k: frame_labels(int(lab.at[k, "onset"]), len(by["train"][k]) * stride).y[::stride] for k in keys}
    gru = SupervisedGRU(seed=a.seed).fit({k: by["train"][k] for k in keys[:cut]}, std, tgt)
    gru_cal = [k for k in keys[cut:] if lab.at[k, "label"] == "nominal"]

    sm = lambda s: causal_ema(s, 0.5)  # noqa: E731
    lp_cache: dict = {}

    def lp_all(k, f):
        if k not in lp_cache:
            lp_cache[k] = lp.score_all(f)
        return lp_cache[k]

    MON = {
        "latent_unc": lambda k, f: sm(lp_all(k, f)["uncertainty"]),
        "latent_err": lambda k, f: sm(lp_all(k, f)["error"]),
        "knn": lambda k, f: sm(knn.score(f)),
        "velocity": lambda k, f: sm(vel.score(f)),
        "supervised": lambda k, f: sm(gru.score(f)),
    }
    cal_ids = {m: (gru_cal if m == "supervised" else train_nom) for m in MON}
    mx = lambda s: r(np.nanmax(s)) if np.isfinite(s).any() else None  # noqa: E731

    calib = {m: sorted(x for x in (mx(fn(k, feats[k])) for k in cal_ids[m]) if x is not None) for m, fn in MON.items()}
    test_ids = sorted(by["test"])
    test = [{"id": k, "label": lab.at[k, "label"], "item": lab.at[k, "sublabel"],
             "max": {m: mx(fn(k, feats[k])) for m, fn in MON.items()}} for k in test_ids]
    print("scored", len(test), "test clips", flush=True)

    # player clips, chosen by rule so the demo shows hits and misses at the default 5% setting
    def first_alarm(m, k):
        th = np.quantile(calib[m], 0.95)
        hit = np.flatnonzero(np.nan_to_num(MON[m](k, feats[k]), nan=-np.inf) > th)
        return hit[0] * stride if hit.size else -1

    def after_onset(m, k):
        o, fa = int(lab.at[k, "onset"]), first_alarm(m, k)
        return fa >= o and fa - o <= 1.5 * FPS

    late = sorted((t["id"] for t in test if t["label"] != "nominal" and 15 <= lab.at[t["id"], "onset"] <= 90),
                  key=lambda k: -lab.at[k, "onset"])
    wm = [k for k in late if after_onset("latent_unc", k)][:2]
    nn = [k for k in late if k not in wm and after_onset("knn", k) and first_alarm("latent_unc", k) < 0][:1]
    PLAYER = wm + nn + [MISSED, NORMAL]
    print("player clips", PLAYER, flush=True)
    vdir = data / "vjepa2_scores"
    vcal = sorted(x for x in (r(np.nanmax(causal_ema(hold_last(np.load(f)["max"]), 0.5)))
                              for f in vdir.glob("*.npz") if lab.at[f.stem, "split"] == "train"))[:15] if vdir.exists() else []
    player = []
    for k in PLAYER:
        fr = read_frames(data / "videos" / f"{k}.mp4", stride=stride)
        for i, f in enumerate(fr):
            save_jpg(f, img / "clips" / k / f"{i:03d}.jpg")
        series = {m: [r(x) for x in fn(k, feats[k])] for m, fn in MON.items()}
        if (vdir / f"{k}.npz").exists():
            s = causal_ema(hold_last(np.load(vdir / f"{k}.npz")["max"]), 0.5)[::stride]
            series["vjepa2"] = [r(x) for x in s]
        o = int(lab.at[k, "onset"])
        player.append({"id": k, "label": lab.at[k, "label"], "item": lab.at[k, "sublabel"], "n": len(fr),
                       "onset_s": None if o < 0 else min(o, len(fr) * stride - 1) / FPS, "series": series})

    # defect wall
    defects = [t for t in test if t["label"] != "nominal"]
    pick = np.random.default_rng(7).choice(len(defects), min(WALL_N, len(defects)), replace=False)
    wall = []
    for j in sorted(pick):
        t = defects[j]
        f = read_frames(data / "videos" / f"{t['id']}.mp4")
        o = min(int(lab.at[t["id"], "onset"]), len(f) - 1)
        shots = {"before": max(o - 24, 0), "onset": o, "after": min(o + 24, len(f) - 1), "end": len(f) - 1}
        for tag, i in shots.items():
            save_jpg(f[i], img / "wall" / t["id"] / f"{tag}.jpg", quality=70)
        wall.append({**t, "onset_s": o / FPS, "t": {tag: i / FPS for tag, i in shots.items()}})

    on = idx.loc[idx.onset >= 0, "onset"].to_numpy() / FPS
    edges = np.arange(0, 3.01, 0.1)
    h, _ = np.histogram(np.clip(on, 0, 2.999), edges)
    res = {}
    for fn in ("pilot_baselines.json", "pilot_vjepa2_cpu.json"):
        fp = ROOT / "results" / fn
        if fp.exists():
            for k, v in json.loads(fp.read_text())["results"].items():
                o5 = next(o for o in v["online"] if o["target_fa"] == 0.05)
                res[k] = {"auroc": r(v["clip"]["auroc"], 3), "ci": [r(c, 3) for c in v["clip"].get("auroc_ci95") or []],
                          "n": v["clip"]["n_clips"], "open": r(v["clip"]["auroc_open"], 3),
                          "decon": r(v["clip"]["auroc_deconstruction"], 3), "detect_5": r(o5["detect_rate"], 3),
                          "fa_test_5": r(o5["fa_test"], 3)}
    site = {
        "fps": FPS / stride, "calib": calib, "test": test, "player": player, "wall": wall, "vjepa2_calib": vcal,
        "onset": {"edges_s": edges.round(2).tolist(), "counts": h.tolist(), "n": int(len(on)),
                  "median_s": r(np.median(on), 3), "share_le_0_5s": r(np.mean(on <= 0.5), 3),
                  "share_ge_1s": r(np.mean(on >= 1.0), 3)},
        "counts": {"videos": int(len(idx)), "defects": int((idx.label != "nominal").sum()),
                   "test_clips": len(test), "test_defects": len(defects)},
        "results": res,
    }
    (out / "data.json").write_text(json.dumps(site, separators=(",", ":")))
    n_img = sum(1 for _ in img.rglob("*.jpg"))
    size = sum(f.stat().st_size for f in img.rglob("*.jpg"))
    print(f"wrote {out / 'data.json'} ({(out / 'data.json').stat().st_size / 1e3:.0f} kB) and {n_img} images ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
