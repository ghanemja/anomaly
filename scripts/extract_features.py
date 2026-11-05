"""Per-frame ResNet-18 embeddings for every streamed video (CPU is fine).

    python scripts/extract_features.py --data data/pilot --labels data/labels --stride 3
writes data/pilot/features_resnet18_s3.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armwm.features import FrameEncoder, save_features  # noqa: E402
from armwm.video import read_frames  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--stride", type=int, default=3, help="keep every Nth frame (3 -> 10 fps)")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    data = Path(a.data)
    ids = [json.loads(l)["pick_id"] for l in (data / "manifest.jsonl").read_text().splitlines() if l.strip()]
    if a.limit:
        ids = ids[: a.limit]
    enc = FrameEncoder(a.arch, threads=a.threads)
    feats, t0 = {}, time.time()
    for i, pid in enumerate(ids):
        try:
            feats[pid] = enc(read_frames(data / "videos" / f"{pid}.mp4", stride=a.stride))
        except Exception as e:  # a corrupt clip should not kill a 1-hour run
            print(f"skip {pid}: {e}", flush=True)
        if i % 200 == 0:
            print(f"{i}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    out = data / f"features_{a.arch}_s{a.stride}.npz"
    save_features(out, feats, a.stride)
    print("wrote", out, len(feats))


if __name__ == "__main__":
    main()
