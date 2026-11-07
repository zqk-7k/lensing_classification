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

## 4. ET results

pending

## 5. Ablations and seeds

pending

## 6. Time-delay statistic

pending

## 7. Point-mass lens-profile robustness

pending

## 8. O4

pending

## 9. Known properties

- **Image arrival time.** `pair_rows` places image 1 at the source time of the ler draw, not at its
  own ler arrival time (the relative delay is exact). Checked on 200 ET 0228 systems: the ratio of the
  injected network S/N from 20 Hz to ler's has 5 to 95 % range 0.58 to 1.32 at the source time and 0.91
  to 1.11 at the image's own arrival time, the same as unlensed events (0.92 to 1.09, the scatter from
  the redrawn spins). So the detection threshold is smeared, not shifted: 9.8 % of second images lie
  below S/N 8 from 20 Hz. They stay in the evaluation set, which makes efficiencies slightly
  conservative; selection functions use the injected S/N.
- **Time-delay span.** T = 1050 days from all 0222 arrival times (signed delays); see the protocol
  addendum.
