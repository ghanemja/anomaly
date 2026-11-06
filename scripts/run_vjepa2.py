"""Score clips with the V-JEPA 2 latent world-model monitor, then evaluate.

GPU recommended (bf16; on a 4-core CPU it is about 2.5 min per clip at --hop 4).
For best fidelity stream full-resolution videos (``stream_subset.py --width 0``);
the 320 px pilot transcodes work but lose detail.

    python scripts/run_vjepa2.py --data data/full --labels data/labels \
        --device cuda --dtype bfloat16 --out results/vjepa2.json

Scores are cached per clip in ``<data>/vjepa2_scores/`` so the run resumes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armwm.labels import load_index  # noqa: E402
from armwm.metrics import causal_ema, evaluate  # noqa: E402
from armwm.video import read_frames  # noqa: E402
from armwm.worldmodels.vjepa2 import VJEPA2Config, VJEPA2Monitor, hold_last  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--labels", default="data/labels")
    p.add_argument("--model", default=VJEPA2Config.model)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--target", type=int, default=4)
    p.add_argument("--hop", type=int, default=2)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--resize", choices=("squash", "pad"), default="squash", help="how to fit 2.3:1 frames to the square input")
    p.add_argument("--max-test", type=int, default=0, help="cap test clips (keeps all defects first); 0 = all")
    p.add_argument("--balanced", type=int, default=0, help="instead: N random test defects + N random test nominals")
    p.add_argument("--max-calib", type=int, default=500)
    p.add_argument("--ema", type=float, default=0.5)
    p.add_argument("--out", default="results/vjepa2.json")
    a = p.parse_args()

    data = Path(a.data)
    have = {json.loads(l)["pick_id"] for l in (data / "manifest.jsonl").read_text().splitlines() if l.strip()}
    idx = load_index(a.labels)
    idx = idx[idx.pick_id.isin(have)].reset_index(drop=True)
    test = idx[idx.split == "test"]
    test = test.sort_values("label", key=lambda s: s == "nominal")  # defects first
    if a.balanced:
        d, n = test[test.label != "nominal"], test[test.label == "nominal"]
        test = pd.concat([d.sample(min(a.balanced, len(d)), random_state=0), n.sample(min(a.balanced, len(n)), random_state=0)])
    elif a.max_test:
        test = test.head(a.max_test)
    calib = idx[(idx.split == "train") & (idx.label == "nominal")].head(a.max_calib)

    cfg = VJEPA2Config(model=a.model, device=a.device, dtype=a.dtype, context=a.context,
                       target=a.target, hop=a.hop, frame_stride=a.frame_stride, resize=a.resize)
    mon = VJEPA2Monitor(cfg)
    cache = data / ("vjepa2_scores" if a.resize == "squash" else f"vjepa2_scores_{a.resize}")
    cache.mkdir(exist_ok=True)

    def scores(pid: str) -> dict:
        f = cache / f"{pid}.npz"
        if not f.exists():
            s = mon.score_clip(read_frames(data / "videos" / f"{pid}.mp4"))
            np.savez(f, mean=s["mean"], max=s["max"])
        z = np.load(f)
        return {k: hold_last(z[k]) for k in ("mean", "max")}

    t0 = time.time()
    todo = list(test.pick_id) + list(calib.pick_id)
    S = {}
    for i, pid in enumerate(todo):
        S[pid] = scores(pid)
        if i % 25 == 0:
            print(f"{i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)

    results = {}
    for kind in ("mean", "max"):
        ts = {p: causal_ema(S[p][kind], a.ema) for p in test.pick_id}
        cs = {p: causal_ema(S[p][kind], a.ema) for p in calib.pick_id}
        results[f"vjepa2_{kind}"] = evaluate(ts, idx, cs, stride=1)
        print(kind, json.dumps(results[f"vjepa2_{kind}"]["clip"]))
    meta = {**vars(a), "n_test": len(test), "n_test_defect": int((test.label != "nominal").sum()), "n_calib": len(calib)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
