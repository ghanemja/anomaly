# Do world-model failure monitors survive real production data?

## Question

A 2026 cluster of papers uses robot world models as failure detectors:
FoMo-FD (surgical robots), FARM (frozen V-JEPA2 internals), ContactGuard
(pre-contact, action-conditioned), the bimanual-failure study (uncertainty beats
raw prediction error), and CureWM (released world models "dream of success" and
often score failures as success). All of them evaluate on lab robots and their
own small datasets, and nearly all condition on robot actions.

ARMBench's defect videos are the opposite setting: about 104k real Amazon
warehouse picks, real failures that are hard to simulate (packaging opening,
items coming apart), frame-level onset labels, video only, and a test split at
the operational defect rate (11.1%). We use it to ask:

1. **Online detection.** Scoring frame by frame, how soon after onset does each
   monitor alarm at a fixed nominal false-alarm rate?
2. **Failure insensitivity at scale.** Does an off-the-shelf world model
   (V-JEPA2, Cosmos) separate real failures from normal motion, or does it
   "dream of success" as CureWM reports? Does fitting on nominal clips help or
   hurt?
3. **Video-only cost.** How much do world-model monitors lose without actions,
   compared with a supervised online classifier on the same frames?
4. **Prediction error vs. uncertainty** as the anomaly signal.

## Data facts that shape the protocol

Measured from the released labels and video (see `armwm/labels.py`):

* Clips are 1280x560 at 30 fps, about 4.5 s (about 140 frames).
* Defect onset comes early. The median first defect event is frame 23
  (0.77 s), and 30% of onsets fall at or before frame 15 (0.5 s). A monitor that
  needs 0.5 s of context cannot warn before most defects, so the headline
  metric is **latency after onset**. True early warning is reported only on the
  subset with at least 1 s of pre-onset video.
* The 84k out-of-split videos are all nominal: a pool for fitting normal-only
  models that never touches the benchmark.
* The archive is one 254 GB gzip stream with no random access. Its member order
  is unrelated to label or split, so the first N GB is a random sample.
  `armwm/stream.py` streams it with parallel range requests (about 20 MB/s
  here, about 3.5 h for the whole archive) and keeps only selected videos.
* Amazon's baseline (`baseline/defect_detection_videos.ckpt`) is a PyTorchVideo
  MViT-B (16x4 frames, 224 px, 2-way sigmoid head, 93.4% validation accuracy). It
  is an offline clip classifier, not an online monitor. The ARMBench paper
  (Table IV) reports it on the same official test split (10,000 nominal clips
  added to the defect test set, hence the 11.1% defect rate): **recall 0.73 at
  13% false-positive rate** overall, 0.69 at 0.23 for open, 0.79 at 0.03 for
  deconstruction. The combined point is directly comparable to our clip-level
  ROC, with one difference to keep in mind: it sees the whole clip, so it is an
  upper reference, not a competitor for an online monitor.

## Protocol

| Role | Data |
|---|---|
| Fit normal-only monitors | out-of-split nominal clips ("extra") |
| Calibrate thresholds | train-split nominal clips |
| Fit supervised reference | 80% of train split (thresholds on the other 20%) |
| Report | official test split, never used for fitting or thresholds |

Metrics (`armwm/metrics.py`):

* clip AUROC / AP (max score over the clip), per defect type, with bootstrap
  95% CI;
* pooled frame AUROC, frames labelled defective from first onset on;
* online, at 1/5/10% nominal false-alarm rate: detection rate, share alarmed
  before onset, share alarmed within 0.5 s / 1 s of onset, median latency,
  clip precision at the test split's 11% defect rate.

## Controls

Every headline claim is paired with the control that could break it
(`armwm/analysis.py`, reported by `scripts/run_baselines.py`):

| Claim | Control |
|---|---|
| "monitor A ties / beats monitor B" | paired bootstrap of the AUROC difference on the same clips, not overlapping per-monitor intervals |
| "alarms before onset are (not) foresight" | time-matched null: for each defect clip, the share of nominal clips that alarm before that clip's onset frame |
| "the monitor detects the defect" | pre-onset AUROC: defect clips scored only on frames before onset, against nominal clips cut at the same frame. Above 0.5 means it reacts to something other than the damage (item type, grasp) or has real foresight. Nominal clips have no item label, so this stands in for a per-item breakdown; per-item AUROC (each item type's defects vs. all nominals) is reported too |
| "the world model is more than motion energy" | `_resid` monitors: per-frame score minus a linear fit on the velocity score (fit on train nominal frames), plus the clip-max correlation with velocity |
| "online numbers compare ranking quality" | detection at a threshold set on the test nominals themselves (diagnostic only), next to the calibrated number, so calibration error and ranking are separated |

## Fixed settings

These were set by hand when the code was written (commit `e0b983d`), before the
first test-split run, and have not been changed since. There is no record of
tuning, but nothing proves it either, so they are frozen here for the GPU run:
any later change gets its own commit and is reported as a new configuration.

| Setting | Value |
|---|---|
| Frame encoder | ImageNet ResNet-18, avg + max pooled layer4 (1024-d) |
| Frame stride | 3 (10 fps) |
| Score smoothing | causal EMA, alpha 0.5 |
| PCA dimension (latent predictor, GRU) | 128 |
| Latent predictor | context 4, 5 members, hidden 512, 15 epochs, lr 1e-3 |
| kNN | k 5, bank 60,000 frames |
| Supervised GRU | hidden 128, 20 epochs, lr 1e-3, 5-fold cross-fit on train |
| V-JEPA2 | ViT-L fpc64-256, frame stride 2, context 8, target 4, hop 2, squash resize |
| False-alarm targets | 1 / 5 / 10%; headline at 5% |
| Early-warning subset | onset at least 1 s into the clip |
| Seed | 0 |

## Monitors

| Monitor | Labels? | Runs on CPU? | Code |
|---|---|---|---|
| Feature velocity (embedding motion energy) | no | yes | `scorers/baselines.py` |
| kNN distance to nominal frames | no | yes | `scorers/baselines.py` |
| Latent predictor ensemble (error / uncertainty / surprise) | no | yes | `scorers/baselines.py` |
| Supervised causal GRU | yes | yes | `scorers/baselines.py` |
| V-JEPA2 latent prediction error (mean / max patch) | no | slow (about 10 s per window, 2.5 min per clip) | `worldmodels/vjepa2.py` |
| Cosmos pixel/latent prediction | no | no | to do on GPU |
| Amazon MViT-B clip classifier | yes | slow | to do: load weights into PyTorchVideo `mvit_base_16x4` |

The latent predictor is a deliberately small world model in ResNet-18 feature
space. It tests the protocol end to end on CPU and gives the "cheap world
model" point for comparison.

## V-JEPA2 implementation note

`VJEPA2Model.forward` in Hugging Face encodes the whole clip and then masks
tokens, so the predictor's context has already attended to the target frames.
That leaks the future. `VJEPA2Monitor` encodes the context frames alone, then
predicts the target tokens and compares them with `layer_norm(encoded window)`,
which matches V-JEPA2's training loss. Without that layer norm, errors are on
the wrong scale (about 1.8 instead of about 0.57).

## Results

See `results/PILOT.md` (CPU pilot on the first 40 GiB of the archive).

## Next steps (GPU)

1. Stream the full archive at full resolution (`--gib 254 --width 0
   --extra-every 20`, about 3.5 h, about 60 GB kept).
2. Run `scripts/run_vjepa2.py --device cuda --dtype bfloat16` on the full test
   split. Sweep context (4/8/16 frames) and the model size (ViT-L/H/g).
   Run both `--resize squash` and `--resize pad` (the 2.3:1 frames distort
   when squashed to the square input).
3. Fit-on-nominal arm (CureWM question): fine-tune the V-JEPA2 predictor on
   extra nominal clips only, then re-score.
4. Add a Cosmos predictor monitor behind the same `score_clip` interface, on a
   stratified subset (all test defects plus about 2k nominal).
5. Port Amazon's MViT-B checkpoint for the offline reference number.
6. Localization figures from the per-patch V-JEPA2 error maps.
