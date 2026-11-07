# Audit of the original 0222/0228 catalogs (2026-11-07)

Scripts in `audit/` (run on the analysis server against `/root/autodl-tmp/qkzhang` and the
revision roots under `/root/autodl-tmp/apjs_rev`).

## S/N of the submitted catalogs (single ET-D arm, `audit/snr_audit.py`)

| 0228 catalog | image 1 median | image 2 median | fainter image median | fainter < 8 | fainter > 50 |
|---|---:|---:|---:|---:|---:|
| SIS | 117 | 100 | 82 | 1.0% | 72% |
| PM | 84 | 76 | 62 | 2.2% | 59% |
| Unlensed events | 39 | | | 3.6% | 37% |

0222 is the same within a few percent. The realistic ET population has fainter-image median
S/N 9.0 and 42% below 8.

## Why the S/N is so high (generator settings; `data_gen_check/backup_original/ET_*_GW_events.py`)

- Impact parameter `y ~ U(0.01, 0.3)`: SIS magnifications are at least 4.3 (median 7.4,
  maximum 101) and the flux ratio is at most 1.86, so the two images are bright near-twins.
  For an SIS lens the multiple-image region is y < 1 with p(y) proportional to y.
- Sources only to z = 1, with no lensing-probability weighting; real lensed sources sit at
  z ~ 2 to 3 for ET.
- Lensed and unlensed events share one unweighted prior and no detection selection, so the
  lensed images (median S/N ~100) and the unlensed events (~40) are different populations.
- Masses uniform 10 to 70 Msun (source frame, m2 > m1 in half the draws); isotropic spins up
  to 0.99; z_l = z_s/2; sigma_v uniform 100 to 500 km/s (affects only the delay, which the
  network never sees).

## What is correct

- Lensing amplification sqrt(|mu|), the type-II Morse phase and the second-image time shift.
- Noise independence: same-row noise of image 1 vs image 2, 0222 vs 0228, SIS vs PM vs
  unlensed has |correlation| <= 0.026 over 8192 samples, the chance level (`audit/noise_audit.py`).
- Background hardness: the fraction of unrelated pairs with detector-frame chirp mass within
  1% is 1.4% in the old 0228 SIS catalog, 1.1% (ler ET) and 0.9% (ler O4) (`audit/mc_audit.py`).
  The old background is not easier than a realistic one in this sense.
- Each segment is standardized, so absolute amplitude is not available to the network.

## Completeness and reproducibility gaps

- `src/generation/` is the two-detector LIGO (H1, L1), 10,000-source, m_max = 100 variant. It
  does not write the stored file names (`*_optimal_SNR_{1,2}.npy`, single channel) and cannot
  reproduce the catalogs; `docs/ARTIFACTS.md` says it does. The ET single-arm generators exist
  only on the server.
- The seed is hard-coded (`n = 6130`, `614`, `61301`); the seeds actually used for 0222 and 0228
  are not recorded anywhere.
- The independence audit compares file hashes only, which would pass with reused noise.
- The "test" twin catalogs (unlensed copies of each image, ~4 GB per array) are generated but
  never read.
- Minor: alignment uses the noiseless signal peak; whitening uses the exact PSD; training
  hard negatives are always (image 1, image 2) while evaluation draws image slots at random.

## The revision generator (`generate_strain_rev.py`)

Fixes the population (ler, PL+P masses, EPL+shear lenses, detection-selected images) but its
spins are aligned only (all tilts 0 or pi, phi_12 = 0, median |a| 0.5): no precession and
larger spins than observed. It stores one detector channel and starts waveforms at 20 Hz.

## Headroom (`audit/fmin_gain.py`, pair_meta.csv)

| Change | S/N gain, median |
|---|---:|
| ET: all three arms instead of the best one | 1.52 |
| ET: waveforms and whitening from 5 Hz instead of 20 Hz (one arm) | 1.56 |
| O4: H1+L1+V1 network instead of the best detector | 1.39 |

Projection only, not a measurement: applying the measured ET efficiency-versus-fainter-image-S/N
curve at P_f = 1e-3 to S/N scaled by 1.5 gives ~13%, and by 2.4 (both ET changes) ~30%, against
5.5% measured with this threshold rule. It assumes efficiency depends on S/N alone and caps at the
top bin (S/N > 40: 88%).
