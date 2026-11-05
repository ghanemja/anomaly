"""Stream selected videos out of the 254 GB ARMBench tar.gz without storing it.

The archive is a single gzip stream, so it cannot be read from the middle:
every run decompresses from byte 0. To make that pass as fast as the network
allows, the compressed bytes are fetched as parallel HTTP range requests and
re-assembled in order, decompressed, and parsed as a streaming tar. Only the
members you select are written to disk, optionally transcoded to a small
low-resolution copy, so a 40 GiB pilot slice needs a few GB of disk.

Because the archive order is unrelated to label or split (see
``labels.archive_order``), streaming the first N bytes gives a random sample of
the whole dataset.
"""
from __future__ import annotations

import io
import json
import logging
import shutil
import subprocess
import tarfile
import threading
import time
import zlib
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator

import requests

from .labels import ARCHIVE_URL

log = logging.getLogger(__name__)


def ranged_chunks(
    url: str,
    start: int = 0,
    stop: int | None = None,
    chunk: int = 16 << 20,
    workers: int = 12,
    retries: int = 6,
) -> Iterator[bytes]:
    """Yield ``url[start:stop]`` in order, fetched by ``workers`` parallel ranges."""
    if stop is None:
        stop = int(requests.head(url, allow_redirects=True, timeout=60).headers["Content-Length"])
    local = threading.local()

    def session() -> requests.Session:
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s

    def fetch(a: int, b: int) -> bytes:
        for attempt in range(retries):
            try:
                r = session().get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=120)
                r.raise_for_status()
                if len(r.content) != b - a:
                    raise IOError(f"short read {len(r.content)} != {b - a}")
                return r.content
            except Exception as e:  # network errors: back off and retry
                wait = 2 ** (attempt + 1)
                log.warning("range %d-%d failed (%s); retry in %ds", a, b, e, wait)
                time.sleep(wait)
        raise IOError(f"range {a}-{b} failed after {retries} attempts")

    offsets = iter(range(start, stop, chunk))
    pending: deque[Future] = deque()
    with ThreadPoolExecutor(workers) as pool:
        def submit_next() -> None:
            a = next(offsets, None)
            if a is not None:
                pending.append(pool.submit(fetch, a, min(a + chunk, stop)))

        for _ in range(workers * 2):
            submit_next()
        while pending:
            data = pending.popleft().result()
            submit_next()
            yield data


class GunzipStream(io.RawIOBase):
    """File-like view of the decompressed bytes of an iterable of gzip chunks."""

    def __init__(self, chunks: Iterable[bytes]):
        self._chunks = iter(chunks)
        self._z = zlib.decompressobj(wbits=31)
        self._buf = bytearray()
        self.compressed_read = 0
        self.eof = False

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while len(self._buf) < len(b) and not self.eof:
            data = next(self._chunks, None)
            if data is None:
                self._buf += self._z.flush()
                self.eof = True
                break
            self.compressed_read += len(data)
            self._buf += self._z.decompress(data)
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        del self._buf[:n]
        return n


def transcode(src: Path, dst: Path, width: int = 320, crf: int = 23) -> int:
    """Downscale to ``width`` px wide, keeping every frame (onsets are frame indices).

    Returns the number of frames written.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"scale={width}:-2",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-an", "-threads", "1", "-fps_mode", "passthrough", str(tmp)],
        check=True,
    )
    tmp.rename(dst)
    return count_frames(dst)


def count_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return int(out.split(",")[0])


def stream_videos(
    out_dir: str | Path,
    want: Callable[[str], bool],
    max_bytes: int | None = None,
    url: str = ARCHIVE_URL,
    width: int | None = 320,
    workers: int = 12,
    transcoders: int = 3,
    progress_every: float = 30.0,
) -> Path:
    """Stream the archive, keeping videos whose pick_id satisfies ``want``.

    Writes ``<out_dir>/videos/<pick_id>.mp4`` (low-res if ``width`` is set,
    original otherwise) and appends one JSON line per video to
    ``<out_dir>/manifest.jsonl``. Videos already in the manifest are skipped, so
    a re-run after an interruption only redoes the transcodes (the gzip stream
    itself always restarts at byte 0).
    """
    out_dir = Path(out_dir)
    vid_dir, raw_dir = out_dir / "videos", out_dir / "raw_tmp"
    vid_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    done = set()
    if manifest.exists():
        done = {json.loads(l)["pick_id"] for l in manifest.read_text().splitlines() if l.strip()}
    lock = threading.Lock()
    stats = {"seen": 0, "kept": 0, "t0": time.time(), "last": 0.0}

    def finish(pick_id: str, raw: Path, nbytes: int) -> None:
        dst = vid_dir / f"{pick_id}.mp4"
        try:
            if width:
                n = transcode(raw, dst, width=width)
                raw.unlink()
            else:
                shutil.move(raw, dst)
                n = count_frames(dst)
        except Exception:
            log.exception("failed on %s", pick_id)
            raw.unlink(missing_ok=True)
            return
        with lock, open(manifest, "a") as f:
            f.write(json.dumps({"pick_id": pick_id, "n_frames": n, "src_bytes": nbytes}) + "\n")

    gz = GunzipStream(ranged_chunks(url, 0, max_bytes, workers=workers))
    jobs: deque[Future] = deque()
    with ThreadPoolExecutor(transcoders) as pool:
        try:
            with tarfile.open(fileobj=io.BufferedReader(gz, 1 << 20), mode="r|") as tar:
                for m in tar:
                    if not (m.isfile() and m.name.endswith(".mp4")):
                        continue
                    stats["seen"] += 1
                    pick_id = Path(m.name).stem
                    if pick_id in done or not want(pick_id):
                        continue
                    raw = raw_dir / f"{pick_id}.mp4"
                    with tar.extractfile(m) as src, open(raw, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    stats["kept"] += 1
                    jobs.append(pool.submit(finish, pick_id, raw, m.size))
                    while len(jobs) > transcoders * 4:  # bound the raw backlog on disk
                        jobs.popleft().result()
                    now = time.time()
                    if now - stats["last"] > progress_every:
                        stats["last"] = now
                        mb = gz.compressed_read / 2**20
                        log.info("%.0f MiB read (%.1f MiB/s), %d videos seen, %d kept",
                                 mb, mb / (now - stats["t0"]), stats["seen"], stats["kept"])
        except (tarfile.ReadError, EOFError, zlib.error) as e:
            if max_bytes is None:
                raise
            log.info("stopped at byte budget (%s)", type(e).__name__)
        for j in jobs:
            j.result()
    shutil.rmtree(raw_dir, ignore_errors=True)
    log.info("done: %d videos seen, %d kept", stats["seen"], stats["kept"])
    return manifest
