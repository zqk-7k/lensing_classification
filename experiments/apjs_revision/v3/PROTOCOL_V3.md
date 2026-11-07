# v3 protocol (fixed before any 0228 score was computed)

Committed before the 0228 manifests of the v3 roots were scored. Nothing below is tuned on 0228.

## Data (`generate_v3.py`, `ler_extra_draw.py`)

- Population: the `ler` draws of the v2 rerun (GWTC-3 Power Law + Peak masses, star-formation rate to
  z = 10, EPL + external-shear galaxy lenses) plus extra draws with the same settings and new seeds.
  Detection is unchanged: every image needs network optimal S/N >= 8 above 20 Hz (ler, no spins);
  the two brightest detected images form the pair (image 1 brighter). Pairs whose images fall within
  one 128 s frame (|delay| < 128 s) are excluded and counted.
- Spins redrawn small and precessing (Beta(2, 6) magnitudes; 66 % near-aligned, 34 % isotropic tilts).
- Detectors: ET = three ET-D arms, waveforms and whitening from 5 Hz; O4 = H1, L1 (aLIGO O4 high)
  and V1 (AdV), from 20 Hz. IMRPhenomXPHM, reference frequency 20 Hz.
- Per event and detector: stream A (full band, 2048 Hz, 2 s around the merger) and stream B
  (< 64 Hz, 128 Hz, 64 s), aligned on the geocentric merger time.
- Storage: float16 whitened noiseless signals; Gaussian noise generated on demand. The 0228 catalog
  additionally stores its noisy data, so the test inputs are fixed on disk.
- Roles: ET 0228 = 5000 lensed systems and 10 000 unlensed events, 0222 = the rest (24 878 and 31 971);
  O4 sizes recorded in `generation_meta_v3.json`. 0222 is split 80/20 into training and validation.

## Models (`v3lib.py`) and recipe (`train_v3.py`)

| Variant | Role |
|---|---|
| `pi` | PI-ResNet v3: all detectors, streams A and B, pairwise interaction. **Primary model.** |
| `v2info` | same network and recipe, loudest detector, stream A only, band above 20 Hz: the information of the submitted setting |
| `early` | early-fusion time-domain CNN (Offermans-style class), same encoders, no pairwise interaction |
| `cqt` | CQT-DeiT v3: constant-Q images of both streams, detectors as RGB, ImageNet DeiT-tiny |
| `pi3A` | ablation, all detectors, stream A only (run if GPU time allows) |

Recipe for all: 40 000 steps of 128 pairs, fresh noise every step, +/-0.125 s jitter, sign flips,
signal-amplitude curriculum 2 -> 1 over the first 25 % of steps, AdamW with cosine decay (1e-4 / 1e-4;
the image model keeps the original CQT-DeiT 5e-5 / 5e-4), seed 42. The checkpoint with the best
validation AUC is the one evaluated.

## Evaluation (unchanged frozen analysis)

- `build_0228_pair_manifests.py` on the v3 root: 30/70 calibration/evaluation split of 0228 sources
  (seed 20260711), 100 000 background pairs per partition (70 % hard within source blocks, 30 % easy),
  10 source blocks.
- `infer_v3.py`: stored 0228 noisy data, evaluation offsets; order-dependent models scored in both
  orders and averaged.
- `analyze_0228_core.py` through `compare_v3.py`: thresholds from calibration background quantiles at
  P_f = 1e-2, 1e-3 (primary), 1e-4; efficiency on evaluation positives; source-block bootstrap.
- Primary comparisons, in order: `pi` vs `v2info` (gain from the added information), `pi` vs `cqt`
  (baseline with the same information), `pi` vs `early` (does the pairwise interaction matter).
- Reported regardless of outcome. Second seeds, O4, and the point-mass set follow the same protocol.
