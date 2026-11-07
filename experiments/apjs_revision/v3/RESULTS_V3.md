# v3 results: all detectors, low-frequency start, fresh noise, compact catalogs

Status: data and storage sections final; result sections are filled as the runs finish.
Protocol: `PROTOCOL_V3.md` (committed before any 0228 score, addendum before any score was read).

## 1. Why v3

The v2 rerun (`../RESULTS_REVISION.md`) put the submitted pipeline on a realistic `ler` population and
found 5.3 % at P_f = 1e-3 at ET. The data audit (`../DATA_AUDIT.md`) showed that the submitted catalogs,
and the v2 layout that copied them, throw away most of what a lensing search has: one ET arm of three,
nothing below 20 Hz although most ET signal-to-noise of a BBH lies at 5 to 20 Hz, a single 24 s segment,
and one fixed noise realisation per event reused for every epoch. v3 keeps the population and the
detection definition and changes only what the network sees and how it is trained.

## 2. Data

| | submitted catalogs | v2 rerun (ET) | v3 (ET) |
|---|---|---|---|
| Population | y ~ U(0.01, 0.3), z <= 1, SIS / PM | ler: GWTC-3 PLP, SFR to z = 10, EPL + shear | same as v2 |
| Detection | none (all injected) | ler, network S/N >= 8 from 20 Hz, two images | same as v2 |
| Detectors kept | 1 ET arm | loudest of 3 arms | all 3 arms |
| Low-frequency start | 20 Hz | 20 Hz | 5 Hz |
| Input | 24 s at 4096 Hz | same | 2 s at 2048 Hz + 64 s below 64 Hz at 128 Hz |
| Spins | isotropic, up to 0.99 | ler (aligned) | Beta(2, 6), mostly near-aligned, precessing |
| Noise in training | one fixed realisation | one fixed realisation | fresh every step |
| Lensed systems 0222 / 0228 | 2 500 / 2 500 | 2 500 / 2 500 | 24 878 / 5 000 |
| Unlensed events 0222 / 0228 | 5 000 / 5 000 | 5 000 / 5 000 | 31 971 / 10 000 |
| S/N the network sees, image 1 / image 2 / unlensed (medians) | 115 / 98 / 40 (one arm) | 14.2 / 9.0 / 9.6 (loudest arm, 20 Hz) | 33.5 / 21.0 / 22.1 (3 arms, 5 Hz) |
| v3 split: loudest arm from 5 Hz; all arms from 20 Hz | | | 22.5 / 13.9 / 14.7; 21.4 / 13.4 / 14.3 |

v3 ET 0228 set, quantiles 10 / 50 / 90 %:

| | 10 % | 50 % | 90 % |
|---|---:|---:|---:|
| Fainter image, network S/N from 5 Hz | 12.4 | 19.8 | 44.8 |
| Louder image, network S/N from 5 Hz | 18.5 | 35.7 | 80.2 |
| Unlensed event, network S/N from 5 Hz | 13.3 | 22.1 | 50.2 |
| Source redshift, lensed / unlensed | 1.51 / 1.00 | 2.71 / 2.04 | 4.72 / 3.66 |
| Flux ratio of the two images | 1.12 | 2.21 | 8.19 |
| abs(time delay), days | 0.81 | 21.0 | 148 |

Lens systems: 3 978 with two images, 121 with three, 901 with four (two brightest detected images
paired). 79 systems were dropped because both images fall within one 128 s frame and 43 because the
detector-frame total mass exceeds 800 Msun.

## 3. Storage

| Data | Before | After | Check |
|---|---:|---:|---|
| Submitted catalogs 0222 + 0228 (SIS, PM, unlensed) | 154 GB | 3.7 GB | re-scored from the compact root: max abs(Δlogit) = 0 on all 202 500 pairs, SIS and PM |
| v2 rerun catalogs, ET + O4 | 35 GB | 3.0 GB | max abs(Δlogit) = 0 on every archived prediction (ET SIS 202 500, ET PM 202 339, O4 SIS 202 500, O4 PM 20 741 pairs) |
| v3 ET (5x more events, 3 detectors, two streams) | — | 8.9 GB | — |
| v3 ET point-mass set | — | pending | — |
| v3 O4 | — | pending | — |

Why it shrinks without losing anything the analysis uses:

- The frozen scripts read only the last 8 192 samples of every 24 s row (the 2 s before and around
  the merger); `../compact_catalogs.py` keeps exactly that window in the original dtype under the
  same file names, so every script runs unchanged and sees bit-identical input.
- v3 stores whitened noiseless signals in float16 (relative error 3e-4, far below the noise) and
  draws Gaussian noise on demand: in training on the GPU, fresh at every step, which is also what
  removes the fixed-noise memorisation the v2 recipe was exposed to. Only the 0228 test set stores
  its noisy data, so the test inputs are fixed on disk.

Not used by any script in this repository (no reference anywhere): the 0123 catalogs (77 GB),
`old_data` (30 GB) and the `test_*` twin arrays inside each catalog directory. Deleting the full
originals is the authors' decision; the compact copies and the generation records make them
unnecessary for every number reported here.

## 4. ET results (corrected evaluation set `data_v3_ET2`)

3 500 evaluation pairs, 100 000 evaluation background pairs (70 % hard, 30 % easy); thresholds from
the separate calibration background; source-block bootstrap intervals (95 %), all from the unchanged
`analyze_0228_core.py`. Seed-42 models.

| Model | AUC | P_f = 1e-2 | **P_f = 1e-3** | P_f = 1e-4 |
|---|---:|---:|---:|---:|
| **PI-ResNet v3** (`pi`) | 0.9954 | 86.8 [85.7, 87.8] | **36.8 [35.1, 38.6]** | 13.4 [12.3, 14.5] |
| Early fusion, Offermans-style (`early`) | 0.9906 | 74.9 [73.7, 76.0] | 22.5 [21.5, 23.6] | 4.0 [3.4, 4.6] |
| CQT-DeiT v3, same information (`cqt`) | 0.9868 | 59.0 [57.8, 60.1] | 17.2 [16.3, 18.4] | 4.8 [4.3, 5.3] |
| Same network, submitted information (`v2info`) | 0.9794 | 53.0 [50.7, 54.9] | 17.0 [15.7, 18.3] | 4.2 [3.7, 4.8] |
| *v2 rerun, frozen recipe (for reference)* | 0.752 | 13.2 | 5.3 [4.3, 6.5] | 1.3 |

Paired differences (same pairs, same thresholds rule; percentage points):

| A − B | ΔAUC | P_f = 1e-2 | **P_f = 1e-3** | P_f = 1e-4 |
|---|---:|---:|---:|---:|
| pi − v2info (what the added information gives) | +0.0160 [+0.0151, +0.0169] | +33.8 [32.2, 35.7] | **+19.9 [18.3, 21.9]** | +9.1 [8.0, 10.3] |
| pi − cqt (same information, image baseline) | +0.0086 [+0.0080, +0.0091] | +27.8 [26.7, 28.8] | **+19.6 [18.2, 21.3]** | +8.6 [7.5, 9.6] |
| pi − early (does the pairwise interaction matter) | +0.0048 [+0.0042, +0.0055] | +11.9 [10.7, 12.9] | **+14.3 [12.8, 15.7]** | +9.4 [8.5, 10.3] |

Achieved false-positive probability on the evaluation background at the 1e-3 threshold: 1.18e-3
(hard 1.29e-3, easy 0.93e-3) for `pi`; every model's calibration is in its `core_results.json`.

Reading it:

- The training recipe alone (fresh noise every step, five times more events, signal curriculum) takes
  the submitted information from 5.3 % to 17.0 % at 1e-3; the added information (all three arms, 5 Hz
  start, 64 s low-frequency stream) doubles it again to 36.8 %.
- With the same information, the image baseline does not use it: CQT-DeiT v3 is at the level of the
  submitted-information network. The pairwise-interaction head beats early fusion of the same encoders
  by 14 points at 1e-3 and by a factor 3 at 1e-4, which is the architectural claim the reviewer asked
  to see separated from Offermans et al.

Efficiency at P_f = 1e-3 against the fainter image's network S/N from 5 Hz (quintiles of the evaluation
positives; `selection_functions_0228_sis.csv`):

| fainter-image S/N | n | pi | early | cqt | v2info |
|---|---:|---:|---:|---:|---:|
| < 14.2 | 675 | 6.7 | 1.5 | 0.0 | 0.0 |
| 14.2 to 17.4 | 673 | 12.0 | 6.4 | 0.0 | 1.3 |
| 17.4 to 22.0 | 654 | 27.7 | 14.5 | 0.8 | 3.7 |
| 22.0 to 31.4 | 699 | 48.5 | 26.3 | 9.2 | 14.7 |
| > 31.4 | 799 | 80.5 | 57.1 | 66.8 | 57.3 |

and against the flux ratio of the two images:

| flux ratio | n | pi | early | cqt | v2info |
|---|---:|---:|---:|---:|---:|
| < 1.31 | 752 | 51.1 | 30.9 | 29.1 | 31.9 |
| 1.31 to 1.87 | 754 | 40.3 | 24.5 | 22.9 | 22.9 |
| 1.87 to 2.93 | 671 | 36.1 | 20.1 | 13.7 | 14.3 |
| 2.93 to 5.06 | 625 | 32.3 | 19.7 | 10.6 | 9.1 |
| > 5.06 | 698 | 22.5 | 16.2 | 7.6 | 4.0 |

## 5. Ablations and seeds

Second seeds (43), same recipe and evaluation set:

| Model | AUC | P_f = 1e-2 | P_f = 1e-3 | P_f = 1e-4 |
|---|---:|---:|---:|---:|
| CQT-DeiT v3, seed 42 | 0.9868 | 59.0 [57.8, 60.1] | 17.2 [16.3, 18.4] | 4.8 [4.3, 5.3] |
| CQT-DeiT v3, seed 43 | 0.9869 | 58.1 [56.7, 59.5] | 19.4 [18.2, 20.7] | 4.1 [3.5, 4.7] |
| PI-ResNet v3, seed 43 | pending | | | |

pi (seed 42) − cqt (seed 43): +17.5 [16.4, 18.8] at 1e-3, +28.7 [27.3, 30.1] at 1e-2, +9.2 [8.3, 10.1] at 1e-4.

Stream-A-only ablation (`pi3A`: all three arms, the 2 s full-band stream only, no 64 s
low-frequency stream), corrected evaluation set:

| Model | AUC | P_f = 1e-2 | P_f = 1e-3 | P_f = 1e-4 |
|---|---:|---:|---:|---:|
| `v2info`: loudest arm, 2 s, above 20 Hz | 0.9794 | 53.0 [50.7, 54.9] | 17.0 [15.7, 18.3] | 4.2 [3.7, 4.8] |
| `pi3A`: three arms, 2 s | 0.9932 | 77.1 [76.5, 77.9] | 31.0 [29.7, 32.4] | 7.5 [6.7, 8.3] |
| `pi`: three arms, 2 s + 64 s below 64 Hz | 0.9954 | 86.8 [85.7, 87.8] | 36.8 [35.1, 38.6] | 13.4 [12.3, 14.5] |

pi − pi3A: +9.7 [8.8, 10.5] at 1e-2, +5.8 [4.3, 7.4] at 1e-3, +5.9 [4.9, 6.8] at 1e-4 (first ET build:
+8.9, +6.5, +4.7). Of the 19.9-point gain over the submitted information at 1e-3, about 14 points come
from the three arms and the band below 20 Hz in the 2 s window, and 6 from the 64 s inspiral stream,
which also nearly doubles the efficiency at 1e-4.

Sensitivity to the evaluation set: the first ET build (image 1 at the ler source time, `work_v3_ET`)
gives, at 1e-3, pi 37.1 [36.1, 38.3], early 22.6, cqt 17.3, v2info 15.4, pi_td 51.2, cqt_td 28.6; paired
differences pi − cqt +19.8 [18.8, 20.9], pi − early +14.6 [13.5, 15.5], pi − v2info +21.7 [20.5, 23.1].
Every number is within 2 points of the corrected set, as expected from the small S/N smear at ET.

Note on `v2info`: it keeps the loudest arm chosen by the noiseless signal energy, an oracle choice that
favours the baseline.

## 6. Time-delay statistic

s_td = logit + ln[p_L(dt) / p_U(dt)] (`combine_td.py`, pre-registered): p_L a KDE in ln dt of the
24 878 training-catalog lensed delays, p_U the difference of two uniform times over T = 1050 days.
Thresholds again from the calibration background only. Same evaluation set as Section 4.

| Model + time delay | AUC | P_f = 1e-2 | **P_f = 1e-3** | P_f = 1e-4 |
|---|---:|---:|---:|---:|
| **PI-ResNet v3** | 0.9970 | 91.7 [91.0, 92.3] | **51.0 [49.7, 52.2]** | 23.9 [22.7, 25.1] |
| Early fusion | 0.9929 | 82.1 [81.2, 82.9] | 42.5 [41.2, 43.6] | 15.7 [14.3, 17.1] |
| Same network, submitted information | 0.9849 | 66.2 [65.1, 67.3] | 31.6 [29.9, 33.5] | 13.7 [12.5, 14.7] |
| CQT-DeiT v3 | 0.9888 | 70.2 [68.9, 72.1] | 29.3 [28.1, 30.7] | 9.8 [8.9, 10.7] |

| A − B | P_f = 1e-2 | **P_f = 1e-3** | P_f = 1e-4 |
|---|---:|---:|---:|
| pi_td − pi (what the delay adds) | +4.9 [4.2, 5.7] | **+14.1 [12.8, 15.3]** | +10.5 [9.4, 11.7] |
| pi_td − cqt_td | +21.5 [20.0, 22.7] | **+21.6 [20.9, 22.3]** | +14.1 [13.0, 15.3] |
| pi_td − early_td | +9.7 [8.8, 10.6] | **+8.5 [7.5, 9.4]** | +8.2 [7.0, 9.7] |
| pi_td − v2info_td | +25.5 [24.6, 26.5] | **+19.3 [18.5, 20.2]** | +10.2 [9.4, 11.1] |

The delay prior helps every model, and the ordering is unchanged; the network's ranking is
complementary to the delay information (it adds 14 points at 1e-3 on top of the network alone).
Achieved false-positive probability of `pi_td` on the evaluation background at the 1e-3 threshold:
1.00e-3 (hard 1.11e-3, easy 0.73e-3).

## 7. Point-mass lens-profile robustness

The 5 000 ET 0228 sources re-lensed by a point mass (log-uniform 1e8 to 1e10 Msun, p(y) proportional
to y on (0, 1]; `generate_pm_v3.py`); 4 668 accepted (both images at network S/N >= 8 from 20 Hz at
their own arrival times, delay >= 128 s). Its own manifests (3 268 evaluation pairs, 100 000
background pairs drawn from point-mass images and unlensed events), scored by the **galaxy-trained**
seed-42 models: a test of the lens profile, not a target population. Fainter-image network S/N
13.1 / 17.6 / 34.2 (10 / 50 / 90 %), flux ratio 1.38 / 2.52 / 5.39, delay 0.05 / 0.36 / 2.6 days.

| Model | AUC | P_f = 1e-2 | **P_f = 1e-3** | P_f = 1e-4 |
|---|---:|---:|---:|---:|
| **PI-ResNet v3** | 0.9950 | 84.8 [84.0, 85.7] | **33.5 [32.1, 34.9]** | 9.7 [8.8, 10.7] |
| Early fusion | 0.9880 | 70.7 [69.7, 71.6] | 20.7 [19.6, 22.0] | 1.8 [1.3, 2.4] |
| CQT-DeiT v3 | 0.9844 | 49.7 [48.0, 51.7] | 12.9 [11.9, 14.0] | 5.2 [4.3, 6.2] |
| Same network, submitted information | 0.9756 | 43.8 [42.4, 45.1] | 12.7 [11.6, 13.7] | 4.9 [4.1, 5.8] |

| A − B | P_f = 1e-2 | **P_f = 1e-3** | P_f = 1e-4 |
|---|---:|---:|---:|
| pi − cqt | +35.1 [32.7, 37.3] | **+20.5 [18.9, 22.3]** | +4.5 [3.6, 5.4] |
| pi − early | +14.1 [13.0, 15.2] | **+12.8 [11.3, 14.0]** | +7.8 [7.2, 8.5] |
| pi − v2info | +40.9 [39.1, 42.9] | **+20.8 [19.3, 22.4]** | +4.8 [3.9, 5.6] |

Achieved false-positive probability of `pi` at the 1e-3 threshold: 1.21e-3 (hard 1.16e-3, easy
1.33e-3). The galaxy-trained network loses 3 points at 1e-3 on a lens profile it never saw (36.8 to
33.5 %), and the ordering of the models is the same as for galaxy lenses.

## 8. O4

pending

## 8b. Compute cost (`bench_v3.py`, measured 2026-11-08)

PI-ResNet v3 scores a pair from two per-event feature vectors, so a catalogue search encodes each event
once and evaluates only the small pairwise head per pair: cost(N) = N c_event + N(N-1)/2 c_head. The
early-fusion and image models must process every pair in full.

| | GPU, RTX 5000 Ada (bf16, batch 512) | CPU, 1 thread, Xeon Silver 4314 (fp32, batch 16) |
|---|---:|---:|
| PI-ResNet v3, per event encoding | 3.7e-4 s | 0.27 s |
| PI-ResNet v3, per pair head | 2.8e-8 s | 2.5e-5 s |
| PI-ResNet v3, full pair (both encodings) | 7.3e-4 s | 0.55 s |
| Early fusion, full pair | 3.7e-4 s | 0.27 s |
| CQT-DeiT v3, full pair | 2.3e-4 s | 0.037 s |
| **PI-ResNet v3 per pair, O4-like year (200 events)** | 3.7e-6 s | **2.8e-3 s** |
| **PI-ResNet v3 per pair, ET year (1e5 events, 5e9 pairs)** | 3.5e-8 s | **3.1e-5 s** |

An ET year of pairs costs about 3 GPU-minutes or 1.8 CPU-days on one thread with PI-ResNet v3; the
early-fusion and image models would need 1.4e9 and 1.9e8 CPU-seconds (43 and 6 CPU-years). Parameters:
41.3 M (pi), 40.5 M (early), 5.8 M (cqt). Raw numbers: server `apjs_rev/bench/*.json`. The CPU numbers
were taken with the other GPUs training (load average 6.5 on 64 cores).

## 9. Detection definition, image times, and what was corrected

- **Detection is on an observed S/N.** ler/gwsnr draws, for each image, one observed network S/N from a
  noncentral chi-square (2 degrees of freedom per detector, non-centrality = optimal S/N squared;
  IMRPhenomD, no spins, from 20 Hz) and detects above 8. Images with optimal S/N a little below 8 are
  therefore in the population, as in a real search: in ler's own optimal S/N, 7.6 % of second images at
  ET and 20.7 % at O4 are below 8.
- **Image arrival time (found during v3, corrected before any score was read).** `pair_rows` (shared
  with v2) placed image 1 at the ler source time, not at its own ler arrival time (source time plus its
  time delay, which is where ler evaluated detectability). The relative delay was exact; the antenna
  pattern of the injection was not the one ler selected on. Measured on 200 ET 0228 systems, the ratio
  of injected to ler network S/N from 20 Hz had a 5 to 95 % range of 0.58 to 1.32 at the source time and
  0.91 to 1.11 at the arrival time (unlensed events: 0.92 to 1.09, the scatter of the redrawn spins).
  The effect on second images below S/N 8 was 9.8 % against ler's 7.6 % at ET, and 34.2 % against 20.7 %
  at O4. `generate_v3.py --image-time arrival` (default) fixes it.
  - O4 was regenerated in full and retrained.
  - ET: the evaluation set was regenerated (`data_v3_ET2`; same systems, spins, magnifications, delays
    and noise seeds, unlensed set identical). The ET models, trained on the first build, are kept. ET
    results in Section 4 are on the corrected set; the first set is a sensitivity check (Section 5).
  - The v2 rerun (`../RESULTS_REVISION.md`) used the same `pair_rows` and carries the same effect.
- **Time-delay span.** T = 1050 days from all 0222 arrival times (signed delays); see the protocol
  addendum.
