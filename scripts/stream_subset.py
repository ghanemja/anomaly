"""Stream a pilot subset of ARMBench defect videos.

Keeps every official train/test video in the first ``--gib`` GiB of the archive,
plus every ``--extra-every``-th out-of-split (all-nominal) video as a pool for
fitting normal-only models.

    python scripts/stream_subset.py --labels data/labels --out data/pilot --gib 40
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armwm.labels import download_labels, load_index  # noqa: E402
from armwm.stream import stream_videos  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default="data/labels")
    p.add_argument("--out", default="data/pilot")
    p.add_argument("--gib", type=float, default=40.0, help="compressed GiB of the archive to read (254 = all)")
    p.add_argument("--extra-every", type=int, default=5, help="keep 1 in N out-of-split nominal videos (0 = none)")
    p.add_argument("--width", type=int, default=320, help="transcode width; 0 keeps the original 1280x560")
    p.add_argument("--workers", type=int, default=12, help="parallel HTTP range requests")
    p.add_argument("--transcoders", type=int, default=3)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    idx = load_index(download_labels(a.labels))
    split = dict(zip(idx.pick_id, idx.split))
    extra_rank = {pid: i for i, pid in enumerate(idx[idx.split == "extra"].sort_values("archive_order").pick_id)}

    def want(pid: str) -> bool:
        s = split.get(pid)
        if s in ("train", "test"):
            return True
        return s == "extra" and a.extra_every > 0 and extra_rank[pid] % a.extra_every == 0

    max_bytes = None if a.gib >= 254 else int(a.gib * 2**30)
    stream_videos(a.out, want, max_bytes=max_bytes, width=a.width or None,
                  workers=a.workers, transcoders=a.transcoders)


if __name__ == "__main__":
    main()
