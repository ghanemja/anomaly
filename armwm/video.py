"""Decode ARMBench clips to frame arrays."""
from __future__ import annotations

from pathlib import Path

import av
import numpy as np


def read_frames(path: str | Path, stride: int = 1, max_frames: int | None = None) -> np.ndarray:
    """Return every ``stride``-th frame as uint8 ``(T, H, W, 3)`` RGB.

    Element ``i`` is source frame ``i * stride``, so annotation frame indices
    map to ``onset // stride``.
    """
    out = []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(stream)):
            if i % stride:
                continue
            out.append(frame.to_ndarray(format="rgb24"))
            if max_frames and len(out) >= max_frames:
                break
    if not out:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(out)
