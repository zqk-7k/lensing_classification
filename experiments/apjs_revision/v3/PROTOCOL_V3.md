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

## Secondary statistic: time-delay prior (added before any 0228 score, same commit rule)

`combine_td.py`: s_td = logit + ln[p_L(dt) / p_U(dt)] with dt = |arrival-time difference| of the two
events. p_L is a Gaussian KDE in ln dt of the 0222 lensed pairs (bandwidth by Scott's rule);
p_U = 2 (T - dt) / T^2, the difference of two times uniform over the observing span T (taken from the
0222 event times). Thresholds again come from the 0228 calibration background only. This is the
time-delay information every lensing search uses (Haris et al. 2018; PO2.0); it is reported next to,
not instead of, the network alone.

## Addendum, 2026-11-08 02:15 server time (before any 0228 score was read)

ET inference of `cqt` has run, but no 0228 score, efficiency or comparison has been read or computed
from it. This addendum records decisions taken since the protocol commit:

- **Extra runs.** `pi3A` (stream A only, all detectors) and second seeds (43) of `pi` and `cqt` on ET.
  Their predictions are named `<variant>_s43_*`; comparisons `pi_s43` vs `cqt_s43`, `pi_s43` vs `cqt`,
  `pi` vs `cqt_s43`, `pi` vs `pi3A`. Time-delay versions of `pi`, `cqt`, `v2info`, `early`.
- **Point-mass set** (`generate_pm_v3.py`, ET): the 0228 SIS sources re-lensed by a point mass with the
  priors of the v2 robustness set (log-uniform 1e8 to 1e10 Msun; p(y) proportional to y on (0, 1]),
  accepted when both images have network S/N >= 8 from 20 Hz at their own arrival times and the delay
  is at least 128 s. Own manifests (`manifests_pm`, `--families SIS,PM`), scored by the galaxy-trained
  seed-42 models, comparisons `pi` vs `cqt`, `v2info`, `early` with `--family pm`.
- **Time-delay span.** `combine_td.py` takes T from all 0222 arrival times, including second images,
  which gives T = 1050 days (image 1 is the brighter image, so the delay is signed and second images
  lie up to a year before or after the one-year source window). Kept as committed: below 100 days,
  where nearly all lensed delays lie, it changes ln p_U by at most about 0.2 nats against T = 365 days.
- **Known property of the catalogs, recorded, not corrected.** `pair_rows` (shared with v2) places
  image 1 at the source time of the ler draw rather than at that image's own ler arrival time; the
  relative delay is exact. The antenna pattern is therefore evaluated at a shifted sidereal time, and
  the injected network S/N from 20 Hz scatters about the ler value (median ratio 1.00; 5 to 95 % range
  0.64 to 1.57 for image 1). 1.9 % of image 1 and 9.8 % of image 2 in the ET 0228 set lie below S/N 8
  from 20 Hz; the redrawn spins add a smaller scatter (unlensed: 0.92 to 1.09, 6.4 % below 8). These
  pairs stay in the evaluation set; efficiencies are therefore slightly conservative. Selection
  functions are reported against the injected S/N.

## Addendum 2, 2026-11-08 03:57 server time (before any 0228 score was read)

- **Detection definition, stated exactly.** ler/gwsnr detection is `pdet_type="boolean"` with
  `distribution_type="noncentral_chi2"`: one draw of the observed network S/N from a noncentral
  chi-square with 2 degrees of freedom per detector and non-centrality equal to the squared optimal
  S/N (IMRPhenomD, no spins, from 20 Hz), threshold 8, for each of the two images. Images with optimal
  S/N somewhat below 8 are therefore detected (ET 0228: 7.6 % of second images, O4: 20.7 %), as in a
  real search. The first protocol section's "network optimal S/N >= 8" is shorthand for this.
- **Image arrival time corrected.** The first addendum recorded that image 1 sat at the ler source time.
  `generate_v3.py --image-time arrival` (now the default) places it at its own ler arrival time, which is
  where ler evaluated its detectability; injected and ler S/N then agree to 0.86 to 1.12 (5 to 95 %),
  as for unlensed events. Consequences:
  - **O4** is regenerated in full with the corrected times and its training restarted (the first O4
    build, `data_v3_O4_srctime`, and its partial runs are set aside; no O4 0228 score was computed).
  - **ET**: the evaluation set is regenerated with the corrected times (`data_v3_ET2`: same systems,
    spins, magnifications, delays and noise seeds; unlensed events identical). The ET training set and
    all ET models are kept (training data placed at source times: 2 points more second images below
    S/N 8 than ler's own selection, a small widening of the training distribution). **The primary ET
    results are those on `data_v3_ET2`.** Scores on the first ET evaluation set (`work_v3_ET`) are
    reported as a sensitivity check only.
  - The ET point-mass set is unaffected (its acceptance used its own arrival times).

## Addendum 3, 2026-11-08 08:06 server time (before any O4 0228 score was read)

No O4 comparison has been run and no O4 0228 efficiency has been read (only the `cqt` predictions
exist). GPU 2 became free when the second ET PI-ResNet seed finished, so O4 gets the full set of
primary comparisons that ET has:

- **O4 `early`** (seed 42, same recipe), scored and compared as `pi` vs `early`. `chain_v3_O4.sh` had left
  it out only for lack of GPU time.
- **O4 time-delay versions** of `pi`, `cqt`, `v2info`, `early` with `combine_td.py` (T from the O4 0222
  arrival times), comparisons `pi_td` vs `cqt_td`, `v2info_td`, `early_td`, `pi`, as on ET.
