# armwm: world-model failure monitors on ARMBench

Do video world models catch real robot-handling failures, online, on production
data? This repo is the benchmark harness and baselines for that question, built
on Amazon's ARMBench defect videos (about 104k warehouse picks, about 4.2k with
packaging that opens or items that come apart, frame-level onset labels).

See [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) for the motivation,
protocol and next steps, and [`results/PILOT.md`](results/PILOT.md) for the CPU
pilot.

## What is here

| Path | What it does |
|---|---|
| `armwm/labels.py` | Labels, splits and per-frame onsets, with the dataset-card quirks fixed |
| `armwm/stream.py` | Streams chosen videos out of the 254 GB single-gzip archive (parallel ranges, no full download) |
| `armwm/metrics.py` | Clip, frame and online-latency metrics; thresholds calibrated on held-out nominal clips |
| `armwm/scorers/baselines.py` | CPU monitors: embedding velocity, kNN, latent-predictor ensemble, supervised causal GRU |
| `armwm/worldmodels/vjepa2.py` | Causal V-JEPA2 latent-prediction monitor (context encoded separately so the future cannot leak) |
| `scripts/` | `stream_subset.py` → `extract_features.py` → `run_baselines.py` / `run_vjepa2.py` → `report.py` |

## Quick start (CPU)

```bash
pip install -e ".[torch,dev]"
```

```bash
python scripts/stream_subset.py --labels data/labels --out data/pilot --gib 40
```

```bash
python scripts/extract_features.py --data data/pilot --stride 3
```

```bash
python scripts/run_baselines.py --data data/pilot --labels data/labels --out results/pilot_baselines.json
```

```bash
python scripts/report.py results/pilot_baselines.json
```

The 40 GiB pilot keeps about 5.9k videos (all official-split clips in that part
of the archive plus 1 in 5 out-of-split nominal clips), transcoded to 320 px at
the full 30 fps (about 0.5 GB). Streaming took about 30 min at about 20 MB/s;
feature extraction about 80 min on 4 CPU cores.

## On a GPU

```bash
pip install -e ".[vjepa2]"
```

```bash
python scripts/stream_subset.py --labels data/labels --out data/full --gib 254 --width 0 --extra-every 20
```

```bash
python scripts/run_vjepa2.py --data data/full --labels data/labels --device cuda --dtype bfloat16 --out results/vjepa2.json
```

`run_vjepa2.py` caches per-clip scores, so it resumes after interruption.

## Tests

```bash
python -m pytest -q
```

## Data and licence

ARMBench is CC BY 4.0 (Amazon). Cite Mitash et al., *ARMBench* (arXiv:2303.16382)
when using the defect videos. Nothing from the dataset is committed here; the
scripts download labels (about 34 MB) and stream videos on demand.
