# v3 results: all detectors, low-frequency start, fresh noise, compact catalogs

Status: all sections filled (2026-11-08). Protocol addenda 4 and 5 were decided after scores were read; see PROTOCOL_V3.md.
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
  by 14 points at 1e-3 (and by a factor 3 at 1e-4 for this seed; the 1e-4 point depends on the
  training seed, Section 5), which is the architectural claim the reviewer asked to see separated
  from Offermans et al.

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
| PI-ResNet v3, seed 42 | 0.9954 | 86.8 [85.7, 87.8] | 36.8 [35.1, 38.6] | 13.4 [12.3, 14.5] |
| PI-ResNet v3, seed 43 | 0.9942 | 79.2 [78.0, 80.4] | 35.7 [34.5, 36.7] | 2.2 [1.8, 2.6] |

Paired differences for every seed pairing (percentage points):

| A − B | P_f = 1e-2 | **P_f = 1e-3** | P_f = 1e-4 |
|---|---:|---:|---:|
| pi (42) − cqt (42) | +27.8 [26.7, 28.8] | **+19.6 [18.2, 21.3]** | +8.6 [7.5, 9.6] |
| pi (42) − cqt (43) | +28.7 [27.3, 30.1] | **+17.5 [16.4, 18.8]** | +9.2 [8.3, 10.1] |
| pi (43) − cqt (42) | +20.2 [18.6, 21.8] | **+18.4 [17.1, 19.8]** | −2.6 [−3.4, −1.8] |
| pi (43) − cqt (43) | +21.1 [19.1, 23.0] | **+16.3 [15.2, 17.3]** | −2.0 [−2.8, −1.1] |

The two PI-ResNet seeds agree at the primary operating point (36.8 and 35.7 % at 1e-3; achieved
false-positive probability 1.18e-3 and 0.99e-3), and PI-ResNet beats the image baseline by 16 to 20
points at 1e-3 and by 20 to 29 points at 1e-2 in all four pairings. They differ by 8 points at 1e-2,
and at 1e-4 they disagree outright (13.4 against 2.2 %). The 1e-4 threshold is the tenth-highest of the
100 000 calibration background scores. Seed 43 gives ten calibration background pairs logits of 9.2
to 12.7, above about 98 % of the lensed pairs, while the highest background logit of seed 42 is 7.6.
Six of the ten are "hard" background (images of two different sources from the same source block),
four pair a lensed image with an unrelated unlensed event; their network S/N (12 to 97) is ordinary
for the set, and no single event dominates (the most frequent appears in 4 of the top 60). The source-block bootstrap does not include this training-seed
variance. **Efficiencies at 1e-4 are therefore not a stable property of the recipe**: they are
reported, but no claim rests on them (protocol addendum 4). The CQT-DeiT seeds agree at every point
(4.8 and 4.1 % at 1e-4).

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

Second PI-ResNet seed with the time delay (`pi_s43_td`, protocol addendum 4): AUC 0.9959, 86.2
[85.3, 87.2] at 1e-2, **47.0 [45.8, 48.2] at 1e-3**, 11.5 [10.0, 12.7] at 1e-4 (achieved false-positive
probability 0.85e-3 at the 1e-3 threshold). pi_s43_td − pi_s43: +7.0, **+11.4 [10.3, 12.5]**, +9.3;
pi_s43_td − cqt_td: +16.0 [14.5, 17.4], **+17.7 [17.0, 18.4]**, +1.7 [0.8, 2.5]. The delay prior does
not remove the seed-43 background outliers (8 of its 10 highest calibration background pairs stay in
the top 10, at logits 9.3 to 12.8); it lifts enough lensed pairs past them that at 1e-4 the second seed
goes from 2.2 to 11.5 %, again above the image baseline but still half of seed 42 (23.9 %).

Protocol addendum 5 found a defect in how the statistic handles pairs with dt above the training span
T; it does not trigger at ET (largest dt 708 days against T = 1050 days). The corrected statistic, with
T from the searched catalogue's own event times (718 days), changes no ET efficiency in this section
by more than 0.3 points (`work_v3_ET2/compare_tdc`). It matters at O4 (Section 8a).

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

## 7b. Score precision (fp32 rescoring, protocol addendum 4)

The registered scores were computed under bf16 autocast, which rounds the logits (2 100 to 12 500
distinct values over 205 000 pairs, depending on the model) and creates ties at the thresholds. Every ET
model was rescored in full fp32 precision (`infer_v3.py --precision fp32`: 144 000 to 201 000 distinct
values; |Δlogit| median 0.004 to 0.044, 99th percentile 0.05 to 0.24) and run through the same analysis
(`work_v3_ET2/compare_fp32`).

| Model | P_f = 1e-2, bf16 → fp32 | **P_f = 1e-3** | P_f = 1e-4 | achieved FPP at 1e-3, fp32 |
|---|---:|---:|---:|---:|
| PI-ResNet v3 (seed 42) | 86.8 → 86.9 | **36.8 → 36.1** | 13.4 → 12.5 | 1.13e-3 |
| PI-ResNet v3 (seed 43) | 79.2 → 79.2 | **35.7 → 35.7** | 2.2 → 2.1 | 0.98e-3 |
| Early fusion | 74.9 → 74.9 | **22.5 → 22.3** | 4.0 → 3.9 | 0.92e-3 |
| CQT-DeiT v3 (seed 42) | 59.0 → 58.3 | **17.2 → 16.8** | 4.8 → 4.1 | 1.05e-3 |
| CQT-DeiT v3 (seed 43) | 58.1 → 58.1 | **19.4 → 19.1** | 4.1 → 4.1 | 1.23e-3 |
| Submitted information | 53.0 → 52.7 | **17.0 → 17.0** | 4.2 → 4.2 | 0.96e-3 |
| Stream A only (`pi3A`) | 77.1 → 77.1 | **31.0 → 30.5** | 7.5 → 7.6 | 0.74e-3 |
| PI-ResNet v3 + time delay | 91.7 → 91.6 | **51.0 → 51.0** | 23.9 → 23.3 | 1.00e-3 |
| PI-ResNet v3 (seed 43) + time delay | 86.2 → 86.2 | **47.0 → 47.0** | 11.5 → 11.5 | 0.88e-3 |

Paired differences at 1e-3 in fp32: pi − v2info +19.2 [17.7, 21.1], pi − cqt +19.3 [17.9, 20.9],
pi − early +13.9 [12.4, 15.2], pi − pi3A +5.6 [4.1, 7.3], pi_td − pi +14.9 [13.6, 16.0], pi_td − cqt_td
+21.7 [21.0, 22.3], pi_s43 − cqt_s43 +16.6 [15.5, 17.6]. No efficiency and no paired difference moves by more
than 0.9 points (largest: pi at 1e-4, and pi − pi3A at 1e-4); AUCs agree to four decimals. Rounding
does not change any ordering or conclusion; the registered bf16 numbers stay primary.

O4 (`work_v3_O4/compare_fp32`, all five models): 2 700 to 11 800 distinct bf16 values against about
198 000 in fp32, |Δlogit| median 0.009 to 0.015. No efficiency moves by more than 0.4 points (PI-ResNet
at 1e-3, 16.1 → 15.7) and no paired difference by more than 0.5 points; AUCs agree to 2e-4.

## 8. O4

H1 and L1 at aLIGO O4 high sensitivity, V1 at AdV, all from 20 Hz; same population model, detection
rule, recipe and frozen analysis as ET (`data_v3_O4`, `work_v3_O4`). Lensed systems 13 536 (0222) and
4 000 (0228), unlensed events 25 801 and 8 000. 2 145 of 20 000 drawn systems (10.7 %) were dropped
because their two brightest images arrive within one 128 s frame, and 319 because the detector-frame
total mass exceeds 800 Msun. Evaluation: 2 800 lensed pairs and 100 000 background pairs (70 % hard).

| O4 0228 set | 10 % | 50 % | 90 % |
|---|---:|---:|---:|
| Fainter image, network S/N from 20 Hz | 7.1 | 9.5 | 15.0 |
| Louder image, network S/N from 20 Hz | 8.5 | 11.6 | 18.9 |
| Fainter image, loudest detector | 5.0 | 6.8 | 10.8 |
| Unlensed event, network S/N | 7.0 | 9.5 | 16.1 |
| Source redshift, lensed / unlensed | 0.97 / 0.23 | 1.80 / 0.61 | 3.11 / 1.31 |
| Flux ratio of the two images | 1.03 | 1.25 | 2.28 |
| abs(time delay), days | 0.01 | 0.40 | 27.9 |

63 % of the lensed pairs come from four-image systems (2 517 of 4 000; 1 106 two-image, 377
three-image), consistent with both images needing magnification to be detected at O4 sensitivity: the
pairs are mostly the two brightest images of quads, nearly equal in flux and close in time.

| Model | AUC | P_f = 1e-2 | **P_f = 1e-3** | P_f = 1e-4 | achieved FPP at 1e-3 |
|---|---:|---:|---:|---:|---:|
| **PI-ResNet v3** (seed 42) | 0.9654 | 43.6 [42.5, 44.8] | **16.1 [15.1, 17.0]** | 6.0 [5.4, 6.8] | 0.94e-3 |
| PI-ResNet v3 (seed 43) | 0.9661 | 44.8 [43.1, 46.4] | **17.5 [16.6, 18.4]** | 5.2 [4.4, 6.0] | 1.01e-3 |
| Same network, submitted information (`v2info`) | 0.9368 | 31.6 [30.2, 33.2] | 12.9 [12.1, 13.7] | 3.5 [2.9, 4.1] | 1.36e-3 |
| CQT-DeiT v3 (`cqt`) | 0.8754 | 23.9 [22.8, 25.0] | 10.2 [9.7, 10.8] | 4.7 [4.0, 5.5] | 1.37e-3 |
| Early fusion (`early`) | 0.9167 | 20.9 [19.6, 22.1] | 5.8 [5.0, 6.6] | 1.7 [1.3, 2.1] | 1.04e-3 |
| *v2 rerun (warm-started from ET; the frozen cold starts collapsed)* | 0.696 | 8.4 | 1.9 [1.5, 2.5] | 0.34 | |

| A − B | ΔAUC | P_f = 1e-2 | **P_f = 1e-3** | P_f = 1e-4 |
|---|---:|---:|---:|---:|
| pi − v2info | +0.0286 [+0.0250, +0.0323] | +12.1 [11.2, 12.9] | **+3.2 [2.2, 3.9]** | +2.5 [1.7, 3.5] |
| pi − cqt | +0.0900 [+0.0849, +0.0946] | +19.8 [18.7, 20.9] | **+5.9 [4.9, 6.8]** | +1.3 [0.5, 2.3] |
| pi − early | +0.0487 [+0.0463, +0.0511] | +22.8 [21.9, 23.8] | **+10.3 [9.2, 11.5]** | +4.4 [3.7, 5.3] |
| pi (43) − cqt | +0.0907 [+0.0857, +0.0953] | +20.9 [19.1, 22.7] | **+7.3 [6.4, 8.2]** | +0.5 [−0.3, 1.4] |
| pi (43) − v2info | +0.0293 [+0.0258, +0.0332] | +13.2 [11.6, 14.9] | **+4.6 [3.3, 6.0]** | +1.7 [0.8, 2.6] |

Reading it:

- The O4 PI-ResNet now trains from scratch: both seeds converge (AUC 0.965 and 0.966) where all three
  v2 cold starts collapsed (validation AUC about 0.52), and the seeds agree at every operating point
  (16.1 and 17.5 % at 1e-3; 6.0 and 5.2 % at 1e-4). Against the v2 warm-started run the efficiency at
  1e-3 rises from 1.9 to 16.1 % (the v2 O4 catalogs also carried the image-time bias of Section 9, 34 %
  of second images below S/N 8 against ler's 21 %).
- PI-ResNet is first at every operating point and every pairing; the margins are smaller than at ET
  (3 to 7 points at 1e-3 over the submitted-information network and the image baseline, 10 over early
  fusion). The information added at O4 is smaller than at ET: every model already starts at 20 Hz, and
  what is added is the other two detectors and the 20 to 64 Hz inspiral stream.
- `v2info` and `cqt` run slightly above the nominal false-positive rate at 1e-3 (1.36e-3 and 1.37e-3
  against 0.94e-3 for PI-ResNet), which favours them in the efficiencies above.
- At 1e-4 PI-ResNet's margin over the image baseline is 0.5 to 1.3 points, at the edge of the
  intervals.

Efficiency at P_f = 1e-3 against the fainter image's network S/N (quintiles of the calibration
positives) and against the flux ratio:

| fainter-image S/N | n | pi | pi (43) | early | cqt | v2info |
|---|---:|---:|---:|---:|---:|---:|
| < 7.8 | 552 | 2.9 | 3.8 | 0.7 | 1.1 | 1.6 |
| 7.8 to 8.9 | 524 | 4.8 | 5.5 | 0.2 | 1.1 | 3.6 |
| 8.9 to 10.0 | 514 | 5.8 | 7.8 | 1.9 | 2.3 | 5.8 |
| 10.0 to 12.0 | 591 | 11.8 | 13.7 | 2.5 | 5.4 | 10.5 |
| > 12.0 | 619 | 49.9 | 51.5 | 21.3 | 37.2 | 38.9 |

| flux ratio | n | pi | pi (43) | early | cqt | v2info |
|---|---:|---:|---:|---:|---:|---:|
| < 1.07 | 496 | 24.2 | 24.8 | 8.9 | 14.7 | 19.2 |
| 1.07 to 1.17 | 596 | 21.0 | 23.3 | 6.9 | 13.9 | 16.3 |
| 1.17 to 1.34 | 602 | 16.6 | 18.3 | 5.3 | 9.8 | 12.0 |
| 1.34 to 1.75 | 575 | 12.7 | 13.9 | 4.5 | 9.0 | 11.0 |
| > 1.75 | 531 | 6.0 | 7.2 | 3.6 | 3.6 | 6.4 |

As in v2, the O4 efficiency lives in the fainter image's S/N: half of the pairs whose fainter image is
above network S/N 12 are recovered at 1e-3, and only 3 to 12 % below it.

### 8a. O4 with the time-delay statistic

The registered statistic is defective at O4 (protocol addendum 5): it takes the span T from the
training catalogue (617 days), the O4 evaluation catalogue spans 692 days, and the 26 background
pairs with dt > T get the largest log ratio of all (+13.0 nats) and set the 1e-4 thresholds. The
corrected statistic takes T from the searched catalogue's own event times (no labels; every pair has
dt <= T). On ET the correction changes no efficiency by more than 0.3 points (`compare_tdc`).

| Model + time delay | registered: 1e-2 / **1e-3** / 1e-4 | corrected: AUC | 1e-2 | **1e-3** | 1e-4 |
|---|---|---:|---:|---:|---:|
| **PI-ResNet v3** | 83.0 / **62.9** / 8.9 | 0.9914 | 83.3 [81.7, 84.6] | **64.5 [62.4, 66.6]** | 47.2 [45.6, 48.8] |
| PI-ResNet v3 (seed 43) | 82.5 / **62.4** / 11.6 | 0.9912 | 82.9 [81.4, 84.4] | **64.2 [62.0, 66.7]** | 49.8 [47.9, 51.9] |
| Same network, submitted information | 77.2 / **57.5** / 6.5 | 0.9856 | 77.5 [76.2, 78.8] | **61.0 [58.6, 63.3]** | 44.8 [43.1, 46.6] |
| Early fusion | 75.4 / **54.4** / 2.3 | 0.9830 | 75.5 [73.9, 77.1] | **56.7 [54.4, 58.9]** | 38.8 [37.6, 39.9] |
| CQT-DeiT v3 | 72.4 / **50.9** / 0.0 | 0.9743 | 72.8 [70.9, 74.7] | **53.7 [52.2, 55.4]** | 33.5 [32.5, 34.7] |

Corrected paired differences: pi_tdc − pi +39.7 / **+48.4 [46.5, 50.7]** / +41.1; pi_tdc − cqt_tdc
+10.5 / **+10.8 [9.8, 12.0]** / +13.6; pi_tdc − early_tdc +7.8 / **+7.8 [7.0, 8.8]** / +8.4;
pi_tdc − v2info_tdc +5.8 / **+3.5 [2.3, 4.8]** / +2.4; pi_s43_tdc − cqt_tdc +10.1 / **+10.5 [9.3, 11.6]** /
+16.3. Achieved false-positive probability of `pi_tdc` at the 1e-3 threshold 1.07e-3.

At O4 the time delay carries most of the information: the lensed delays are short (median 0.4 days)
against a 692-day span, and the prior quadruples PI-ResNet's efficiency at 1e-3 (16.1 to 64.5 %).
With the delay included in every model, PI-ResNet still leads every baseline by 3.5 to 10.8 points at
1e-3 (both seeds). Two caveats belong next to these numbers: p_L is estimated from the same simulated
lens population as the test set, so the gain is what a search with the right lens-population model
would get; and the 128 s minimum delay leaves out the 10.7 % of systems whose two brightest images
overlap in time (a separate analysis problem), both from the test positives and from p_L.

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
