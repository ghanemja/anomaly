"""ARMBench defect-video labels.

Loads the official train/test split and per-frame event annotations from the
`amazon/armbench-defect-video` Hugging Face repo, and fixes the quirks listed on
its dataset card:

* event label strings carry trailing spaces ("Open ", "Deconstruction ");
* some ``frame_idx`` values exceed the clip's real frame count (clamp, don't shift);
* ``defect_onset_frames`` omits some events, so onset is taken over all
  defect events (``onset_frames_all_events``);
* the 84k videos outside the official split are all nominal and form a pool
  for fitting normal-only models.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HF_REPO = "amazon/armbench-defect-video"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
ARCHIVE_URL = f"{HF_BASE}/archive/armbench-defects-video-0.1.tar.gz"
ARCHIVE_BYTES = 272_781_053_674

LABEL_FILES = {
    "labels_train.parquet": "labels/train-00000-of-00001.parquet",
    "labels_test.parquet": "labels/test-00000-of-00001.parquet",
    "video_annotations.jsonl": "annotations/video_annotations_104075.jsonl",
    "archive_listing.txt": "annotations/archive_listing_verbose.txt",
}

DEFECT_EVENTS = ("open", "deconstruction")
CLASSES = ("nominal", "open", "deconstruction")
FPS = 30.0  # measured: nearly every clip is 30/1; a few report ~30.1


def download_labels(dest: str | os.PathLike, overwrite: bool = False) -> Path:
    """Fetch the ~34 MB of label/annotation files (never the 254 GB archive)."""
    import requests

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for local, remote in LABEL_FILES.items():
        out = dest / local
        if out.exists() and not overwrite:
            continue
        with requests.get(f"{HF_BASE}/{remote}", stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = out.with_suffix(out.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.rename(out)
    return dest


def _event_frames(annotations: list[dict]) -> dict[str, list[int]]:
    events: dict[str, list[int]] = {k: [] for k in DEFECT_EVENTS}
    for a in annotations:
        if a.get("frame_idx") is None:
            continue
        lab = str(a.get("label", "")).strip().lower()
        if lab in events:
            events[lab].append(int(float(a["frame_idx"])))
    return {k: sorted(v) for k, v in events.items()}


def load_index(label_dir: str | os.PathLike) -> pd.DataFrame:
    """One row per video in the archive (104,075), with split, class and onset.

    Columns: pick_id, split ('train' | 'test' | 'extra'), label, sublabel,
    onset_open, onset_deconstruction, onset (first defect event of any type,
    -1 for nominal), plus archive_order / archive_end_offset when the archive
    listing is present.
    """
    label_dir = Path(label_dir)
    rows = []
    with open(label_dir / "video_annotations.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ev = _event_frames(r.get("annotations") or [])
            sub = r.get("sublabel")
            rows.append(
                {
                    "pick_id": r["id"],
                    "label": r["label"].strip().lower(),
                    "sublabel": sub if isinstance(sub, str) else "",
                    "onset_open": ev["open"][0] if ev["open"] else -1,
                    "onset_deconstruction": ev["deconstruction"][0] if ev["deconstruction"] else -1,
                }
            )
    df = pd.DataFrame(rows)
    on = df[["onset_open", "onset_deconstruction"]].replace(-1, np.iinfo(np.int64).max).min(axis=1)
    df["onset"] = np.where(on == np.iinfo(np.int64).max, -1, on).astype(int)

    split = pd.concat(
        [pd.read_parquet(label_dir / p, columns=["pick_id", "split"]) for p in ("labels_train.parquet", "labels_test.parquet")]
    ).set_index("pick_id")["split"]
    df["split"] = df["pick_id"].map(split).fillna("extra")

    listing = label_dir / "archive_listing.txt"
    if listing.exists():
        order = archive_order(listing)
        df = df.merge(order, on="pick_id", how="left")
    return df.reset_index(drop=True)


_VIDEO_RE = re.compile(r"(?:\./)?videos/(\d+)\.mp4$")


def archive_order(listing_path: str | os.PathLike) -> pd.DataFrame:
    """Position of every video in the (sequential, unseekable) tar.gz.

    The offset is the *uncompressed* tar offset where the member ends. mp4 does
    not compress, so it is within ~0.5% of the compressed offset; use it to pick
    how many bytes of the archive to stream for a given subset.
    """
    rows, off, k = [], 0, 0
    with open(listing_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            size, name = int(parts[2]), parts[-1]
            off += 512 + (size + 511) // 512 * 512
            m = _VIDEO_RE.match(name)
            if m:
                rows.append((m.group(1), k, off, size))
                k += 1
    return pd.DataFrame(rows, columns=["pick_id", "archive_order", "archive_end_offset", "mp4_bytes"])


@dataclass(frozen=True)
class FrameLabels:
    """Per-frame ground truth for one clip, clamped to its real length."""

    y: np.ndarray  # 1 from first defect event onward, else 0
    onset: int  # -1 for nominal


def frame_labels(onset: int, n_frames: int) -> FrameLabels:
    """A defect (opened packaging, item in pieces) persists once it starts."""
    y = np.zeros(n_frames, dtype=np.int8)
    if onset >= 0:
        onset = min(onset, n_frames - 1)  # dataset-card quirk 2: clamp
        y[onset:] = 1
    return FrameLabels(y=y, onset=onset)
