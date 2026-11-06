# CPU pilot: first 40 GiB of the archive

**Data.** The first 40 GiB of the archive (16.4k of 104k videos, a random sample
since archive order is unrelated to labels), transcoded to 320x140 at 30 fps.
That gives 1,857 official test clips (217 defects, 11.7%), 1,418 train clips
(950 nominal used for threshold calibration) and 2,630 out-of-split nominal
clips for fitting normal-only monitors.

**Features.** Frozen ImageNet ResNet-18, average- and max-pooled layer4
(1024-d), every 3rd frame (10 fps). Scores are smoothed with a causal EMA
(alpha 0.5). Thresholds come from train-split nominal clips, never from test.

## Results on the test split

`python scripts/report.py results/pilot_baselines.json`; online columns are at
a 5% target nominal false-alarm rate.

| monitor | clip AUROC [95% CI] | clip AP | AUROC open | AUROC decon. | frame AUROC | test FA @5% | detected | alarm before onset | alarm <=1 s after onset | median latency (s) | precision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| velocity | 0.605 [0.57, 0.64] | 0.194 | 0.566 | 0.641 | 0.496 | 4% | 12% | 6% | 5% | 0.52 | 30% |
| knn | 0.633 [0.59, 0.67] | 0.269 | 0.582 | 0.679 | 0.619 | 4% | 16% | 3% | 4% | 1.68 | 38% |
| latent_pred_error | 0.679 [0.64, 0.72] | 0.241 | 0.594 | 0.755 | 0.573 | 4% | 15% | 3% | 7% | 0.70 | 32% |
| latent_pred_uncertainty | 0.644 [0.60, 0.69] | 0.289 | 0.552 | 0.727 | 0.626 | 4% | 23% | 4% | 11% | 0.77 | 43% |
| latent_pred_surprise | 0.519 [0.48, 0.56] | 0.119 | 0.506 | 0.532 | 0.502 | 6% | 4% | 1% | 1% | 1.72 | 8% |
| supervised_gru | 0.692 [0.65, 0.73] | 0.319 | 0.574 | 0.799 | 0.734 | 7% | 31% | 1% | 6% | 1.43 | 38% |

Detection rate at other false-alarm targets:

| monitor | 1% FA | 5% FA | 10% FA |
|---|---|---|---|
| velocity | 4% | 12% | 17% |
| knn | 8% | 16% | 24% |
| latent_pred_error | 4% | 15% | 28% |
| latent_pred_uncertainty | 9% | 23% | 34% |
| latent_pred_surprise | 0% | 4% | 10% |
| supervised_gru | 15% | 31% | 43% |

## What the pilot says

1. **Calibration transfers.** Thresholds set on train-split nominal clips give
   4-7% false alarms on test nominals against a 5% target (1% and 8-13% at the
   1% and 10% targets). The online numbers are therefore honest.
2. **Frozen image features are a weak floor.** The best clip AUROC is 0.69
   (supervised GRU, 95% CI 0.65-0.73). At 5% false alarms the best monitor
   catches about a third of defects. Amazon reports 93% validation accuracy for
   its MViT-B, but that is accuracy on a defect-enriched split, not AUROC at the
   operational rate, so it is not comparable.
3. **A normal-only world model nearly matches supervision.** The latent
   predictor's error (0.68, 95% CI 0.64-0.72) is statistically tied with the
   supervised GRU at clip level, with no defect labels used. Its uncertainty
   variant is the best normal-only *online* monitor (23% detected at 5% FA),
   echoing the bimanual-failure finding that uncertainty beats raw error.
   Dividing error by uncertainty ("surprise") destroys the signal.
4. **"Open" is the hard class.** Every monitor scores 0.50-0.59 on open
   packaging against 0.64-0.80 on items coming apart. Opening is a small, local
   change, and global pooled features barely see it. Patch-level scores (the
   V-JEPA2 `max` variant) and full resolution are the obvious next test.
5. **No early warning yet.** 81 of 217 test defects have at least 1 s of
   video before onset. On those, alarms before onset fire at 2-9%, about the
   nominal false-alarm rate, so they are noise, not foresight.

Caveats: a single 40 GiB sample (217 test defects, so wide intervals), 320 px
transcodes, one seed, no hyperparameter search (deliberately, so there is
nothing to overfit to test).

## V-JEPA2 smoke run (CPU, 30 test clips)

`scripts/run_vjepa2.py --balanced 15 --max-calib 15 --hop 4` on the 320 px
transcodes: 15 random test defects, 15 random test nominals, 15 calibration
clips. About 2.5 min per clip on 4 CPU cores.

| monitor | clip AUROC [95% CI] | AUROC open | AUROC decon. | frame AUROC |
|---|---|---|---|---|
| vjepa2_mean (mean patch error) | 0.39 [0.18, 0.60] | 0.38 | 0.40 | 0.50 |
| vjepa2_max (worst patch error) | 0.55 [0.35, 0.77] | 0.59 | 0.50 | 0.60 |

This is an end-to-end check, not a result: with 30 clips the intervals span
chance. Two things are worth following up at scale. First, the mean error
points *below* chance: failures may be no more surprising to the off-the-shelf
predictor than normal motion, which is CureWM's "dreaming of success" effect.
Second, the worst-patch score is the only variant above 0.5 on open packaging,
the class the global baselines miss. Both need the full test split, at full
resolution, on a GPU.
