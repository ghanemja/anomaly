"""Per-frame ResNet-18 embeddings for every streamed video (CPU is fine).

    python scripts/extract_features.py --data data/pilot --labels data/labels --stride 3
writes data/pilot/features_resnet18_s3.npz. Work is saved in shards of
``--shard`` clips under ``features_resnet18_s3.shards<N>/``, so an interrupted run
resumes where it stopped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armwm.features import FrameEncoder, load_features, save_features  # noqa: E402
from armwm.video import read_frames  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--stride", type=int, default=3, help="keep every Nth frame (3 -> 10 fps)")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", type=int, default=250)
    a = p.parse_args()
    data = Path(a.data)
    ids = [json.loads(l)["pick_id"] for l in (data / "manifest.jsonl").read_text().splitlines() if l.strip()]
    if a.limit:
        ids = ids[: a.limit]
    out = data / f"features_{a.arch}_s{a.stride}.npz"
    shard_dir = out.with_suffix(f".shards{a.shard}")  # shard size in the name: never mix layouts
    shard_dir.mkdir(exist_ok=True)
    enc = None
    t0 = time.time()
    for k in range(0, len(ids), a.shard):
        path = shard_dir / f"{k // a.shard:05d}.npz"
        if path.exists():
            continue
        enc = enc or FrameEncoder(a.arch, threads=a.threads)
        feats = {}
        for pid in ids[k : k + a.shard]:
            try:
                feats[pid] = enc(read_frames(data / "videos" / f"{pid}.mp4", stride=a.stride))
            except Exception as e:  # a corrupt clip should not kill a long run
                print(f"skip {pid}: {e}", flush=True)
        save_features(path, feats, a.stride)
        print(f"{min(k + a.shard, len(ids))}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    merged = {}
    for path in sorted(shard_dir.glob("*.npz")):
        merged.update(load_features(path)[0])
    save_features(out, merged, a.stride)
    print("wrote", out, len(merged))


if __name__ == "__main__":
    main()
