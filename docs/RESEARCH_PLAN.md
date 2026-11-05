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
  is an offline clip classifier, not an online monitor.

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

## Monitors

| Monitor | Labels? | Runs on CPU? | Code |
|---|---|---|---|
| Feature velocity (embedding motion energy) | no | yes | `scorers/baselines.py` |
| kNN distance to nominal frames | no | yes | `scorers/baselines.py` |
| Latent predictor ensemble (error / uncertainty / surprise) | no | yes | `scorers/baselines.py` |
| Supervised causal GRU | yes | yes | `scorers/baselines.py` |
| V-JEPA2 latent prediction error (mean / max patch) | no | slow (6 s per window) | `worldmodels/vjepa2.py` |
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
3. Fit-on-nominal arm (CureWM question): fine-tune the V-JEPA2 predictor on
   extra nominal clips only, then re-score.
4. Add a Cosmos predictor monitor behind the same `score_clip` interface, on a
   stratified subset (all test defects plus about 2k nominal).
5. Port Amazon's MViT-B checkpoint for the offline reference number.
6. Localization figures from the per-patch V-JEPA2 error maps.
