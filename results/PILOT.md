# CPU pilot: first 40 GiB of the archive

**Data.** The first 40 GiB of the archive (16.4k of 104k videos, a random sample
since archive order is unrelated to labels), transcoded to 320x140 at 30 fps.
That gives 1,857 official test clips (217 defects, 11.7%), 1,418 train clips
(950 nominal used for threshold calibration) and 2,630 out-of-split nominal
clips for fitting normal-only monitors.

**Features.** Frozen ImageNet ResNet-18, average- and max-pooled layer4
(1024-d), every 3rd frame (10 fps). Scores are smoothed with a causal EMA
(alpha 0.5). Thresholds come from train-split nominal clips, never from test.
All settings are the fixed ones in `docs/RESEARCH_PLAN.md` ("Fixed settings").

**Reproduce.** `python scripts/run_baselines.py --data data/pilot --labels data/labels`,
then `python scripts/report.py results/pilot_baselines.json`. Per-clip scores
are in `results/pilot_scores.npz`, so every number below and on the site can be
recomputed without re-streaming the videos.

**Changes since the first version of this write-up.** The supervised GRU is now
cross-fitted on the train split (5 folds), so it is calibrated on the same 950
nominal clips as every other monitor, not on about 190. Its clip AUROC moved
from 0.692 to 0.704; the normal-only monitors reproduce exactly. Several claims
in the first version did not survive the controls below and are corrected
here.

## Results on the test split

Online columns are at a 5% target nominal false-alarm rate.

| monitor | clip AUROC [95% CI] | clip AP | AUROC open | AUROC decon. | frame AUROC | test FA @5% | detected | alarm before onset | alarm <=1 s after onset | median latency (s) | precision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| velocity | 0.605 [0.57, 0.64] | 0.194 | 0.566 | 0.641 | 0.496 | 4% | 12% | 6% | 5% | 0.52 | 30% |
| knn | 0.633 [0.59, 0.67] | 0.269 | 0.582 | 0.679 | 0.619 | 4% | 16% | 3% | 4% | 1.68 | 38% |
| latent_pred_error | 0.679 [0.64, 0.72] | 0.241 | 0.594 | 0.755 | 0.573 | 4% | 15% | 3% | 7% | 0.70 | 32% |
| latent_pred_uncertainty | 0.644 [0.60, 0.69] | 0.289 | 0.552 | 0.727 | 0.626 | 4% | 23% | 4% | 11% | 0.77 | 43% |
| latent_pred_surprise | 0.519 [0.48, 0.56] | 0.119 | 0.506 | 0.532 | 0.502 | 6% | 4% | 1% | 1% | 1.72 | 8% |
| supervised_gru | 0.704 [0.66, 0.74] | 0.320 | 0.600 | 0.797 | 0.735 | 6% | 37% | 1% | 3% | 1.83 | 44% |
| latent_pred_error_resid | 0.665 [0.63, 0.71] | 0.229 | 0.570 | 0.751 | 0.636 | 5% | 16% | 2% | 8% | 0.70 | 31% |

`_resid`: the same monitor after removing a linear fit on the velocity score
(see "Motion" below). Residual variants of the other monitors are in the JSON.

Detection rate at each false-alarm target, with the realized test false-alarm
rate, and (in brackets) the diagnostic rate at a threshold set on the test
nominals themselves, so every monitor sits at exactly the target:

| monitor | 1% FA | 5% FA | 10% FA |
|---|---|---|---|
| velocity | 4% at 1% [4%] | 12% at 4% [12%] | 17% at 8% [20%] |
| knn | 8% at 1% [8%] | 16% at 4% [20%] | 24% at 8% [27%] |
| latent_pred_error | 4% at 1% [4%] | 15% at 4% [21%] | 28% at 8% [34%] |
| latent_pred_uncertainty | 9% at 1% [8%] | 23% at 4% [27%] | 34% at 9% [34%] |
| latent_pred_surprise | 0% at 1% [0%] | 4% at 6% [2%] | 10% at 10% [9%] |
| supervised_gru | 9% at 1% [12%] | 37% at 6% [33%] | 43% at 12% [42%] |

## Controls

From `armwm/analysis.py`; the full output is under `controls` and each
monitor's `pre_onset` / `early_warning_null_5` in `pilot_baselines.json`.

Paired clip-AUROC differences (bootstrap on the same resampled clips):

| comparison | difference | 95% CI | p |
|---|---|---|---|
| supervised_gru - latent_pred_error | +0.025 | [-0.026, +0.076] | 0.34 |
| supervised_gru - latent_pred_uncertainty | +0.059 | [+0.007, +0.112] | 0.02 |
| supervised_gru - knn | +0.070 | [+0.015, +0.127] | 0.01 |
| latent_pred_error - latent_pred_uncertainty | +0.034 | [+0.002, +0.068] | 0.03 |
| latent_pred_error - knn | +0.045 | [-0.007, +0.097] | 0.09 |
| latent_pred_error - velocity | +0.073 | [+0.040, +0.107] | <0.001 |
| latent_pred_error - latent_pred_error_resid | +0.013 | [-0.002, +0.028] | 0.08 |

Before the damage, and early alarms against a time-matched null (81 test
defects with at least 1 s of video before onset; 1,640 test nominals):

| monitor | pre-onset AUROC [95% CI] | early alarms at 5% FA: observed vs. chance | p (one-sided) |
|---|---|---|---|
| velocity | 0.60 [0.54, 0.67] | 5 vs. 2.3 | 0.08 |
| knn | 0.51 [0.45, 0.58] | 2 vs. 1.1 | 0.32 |
| latent_pred_error | 0.64 [0.57, 0.70] | 6 vs. 2.4 | 0.03 |
| latent_pred_uncertainty | 0.60 [0.53, 0.67] | 7 vs. 2.6 | 0.02 |
| supervised_gru | 0.57 [0.51, 0.63] | 3 vs. 1.5 | 0.18 |

Per item type (each item's test defects against all test nominals, since
nominal clips carry no item label): latent_pred_error 0.68 on books (126
defects), 0.66 on boxes (76); supervised_gru 0.71 and 0.68.

Reference point: Amazon's MViT-B on the same official test split (ARMBench,
Table IV) reaches **recall 0.73 at 13% false-positive rate**. At a 13% clip-level
false-positive rate our monitors catch 45% (supervised GRU), 37% (uncertainty),
36% (error) and 34% (kNN). MViT-B sees all 32 sampled frames of the clip, so it
is an offline upper reference, not an online competitor, but the gap is large.

## What the pilot says

1. **Calibration transfers.** Thresholds set on train-split nominal clips give
   4-6% false alarms on test nominals against a 5% target (1% and 8-12% at the
   1% and 10% targets).
2. **Frozen image features are a weak floor, far below Amazon's offline
   classifier.** The best clip AUROC is 0.70 (supervised GRU). At Amazon's
   operating point (13% false positives) the best monitor catches 45% of
   defects against MViT-B's 73%.
3. **The normal-only world model matches a *weak* supervised reference on clip
   AUROC, and trails it online.** The difference in clip AUROC is +0.025 for
   the GRU (95% CI -0.026 to +0.076, p = 0.34): not significant, but the
   interval allows a real gap of up to about 0.08. On every other view the
   labels help: AP 0.32 vs. 0.24, and at exactly 5% false alarms 33% vs. 21%
   detected. The GRU itself is small (frozen ResNet-18 features, about 1.1k
   training clips per fold), so this bounds what *these features* gain from
   labels, not what labels can buy.
4. **Error vs. uncertainty is split.** Prediction error ranks clips better
   (clip AUROC +0.034, p = 0.03); ensemble uncertainty gives the better online
   monitor (27% vs. 21% detected at exactly 5% false alarms, AP 0.29 vs. 0.24).
   The first version called this an echo of the bimanual-failure finding
   ("uncertainty beats error"); that only holds for the online operating point.
   Dividing error by uncertainty ("surprise") destroys the signal.
5. **The world model is mostly motion, but its defect signal is not.** On
   nominal frames, 70% of the variance of the predictor's error is explained by
   the velocity (motion-energy) score, and clip-max error correlates 0.66 with
   clip-max velocity. Yet removing the motion fit costs only 0.013 AUROC
   (0.679 to 0.665, p = 0.08), and error beats velocity itself by 0.073
   (p < 0.001). The part that separates defects survives the motion control.
6. **"Open" is the hard class.** Every monitor scores 0.51-0.60 on open
   packaging against 0.64-0.80 on items coming apart, matching Amazon's own
   result (open recall 0.69 at 0.23 FPR vs. 0.79 at 0.03). Patch-level scores
   (the V-JEPA2 `max` variant) and full resolution are the obvious next test.
7. **There is signal before the damage, and it is not yet explained.** The
   first version said early alarms were "luck, not foresight". That compared
   them with the whole-clip false-alarm rate; matched in time, the predictor's
   error and uncertainty alarm before onset 2.5-2.7 times as often as chance
   (6 vs. 2.4, p = 0.03; 7 vs. 2.6, p = 0.02), and defect clips score higher
   than nominal clips *before any damage is annotated* (pre-onset AUROC 0.64
   for error, 0.60 for velocity). Three explanations fit, and the pilot cannot
   tell them apart:
   * a real precursor: picks that end in damage move differently beforehand
     (a fast swing, a poor grasp);
   * a confound: defect picks differ from nominal picks in item type or
     handling, not in the damage itself (nominal clips have no item label, so
     this cannot be checked directly);
   * annotation lag: onset marks the first frame where damage is *clearly
     visible*, and it may begin earlier.

   The appearance-based kNN monitor shows no pre-onset signal (0.51), so if it
   is a confound it lives in the dynamics, not in what the item looks like.
   The counts are small (81 clips, single digits of early alarms) and about ten
   monitors were tested, so none of these p-values survives a Bonferroni
   correction; treat this as a lead for the full split, not a finding. Either
   way, part of every monitor's clip AUROC comes from frames before the damage.

Caveats: a single 40 GiB sample (217 test defects, so wide intervals), 320 px
transcodes, one seed, no hyperparameter search, and the settings were fixed by
hand before the first test run (no record of tuning, but no proof either; they
are frozen for the GPU run).

## V-JEPA2 smoke run (CPU, 30 test clips)

`scripts/run_vjepa2.py --balanced 15 --max-calib 15 --hop 4` on the 320 px
transcodes: 15 random test defects, 15 random test nominals, 15 calibration
clips. About 2.5 min per clip on 4 CPU cores (about 10 s per window). Frames
are squashed from 2.3:1 to the model's square input (`--resize squash`); an
aspect-preserving `--resize pad` is now available and should be compared.

| monitor | clip AUROC [95% CI] | AUROC open | AUROC decon. | frame AUROC |
|---|---|---|---|---|
| vjepa2_mean (mean patch error) | 0.39 [0.18, 0.60] | 0.38 | 0.40 | 0.50 |
| vjepa2_max (worst patch error) | 0.55 [0.35, 0.77] | 0.59 | 0.50 | 0.60 |

This is an end-to-end check, not a result: with 30 clips both intervals
include chance and say nothing about whether V-JEPA2 is better, worse or blind
to failures. Whether an off-the-shelf world model "dreams of success" on this
data (CureWM's question) needs the full test split at full resolution on a
GPU.
