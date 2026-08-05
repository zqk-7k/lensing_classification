# experiment execution log

## Protocol and training

- Archived four original runs as Protocol-v0 pilots after identifying shared-unlensed-pool and image-level SEMD split issues.
- Built the shared source-level 0222 split with seed 42: SIS 2000/500, PM 2000/500, and unlensed 4000/1000 train/validation sources.
- Retrained four frozen v1 checkpoints for 300 epochs and selected checkpoints by validation AUC only.
- Froze checkpoint hashes in `checkpoint_registry.json`.

## Evaluation lock and holdout audit

- Locked the evaluation protocol before held-out score inspection and created a protocol tag.
- Revised 50 source blocks to 10 before inference to permit 100,000 unique within-block background pairs; recorded this as protocol revision 1.1 before inference.
- Audited 0222/0228 file hashes, exact source-row overlap, and sampled waveform hashes. The audit passed; seed metadata was unavailable and is not claimed.

## Shared pairs and inference

- Split 0228 at source level into 30% calibration and 70% evaluation using seed 20260711.
- Generated four shared manifests, each with one positive per source and 100,000 unique background pairs (70% hard, 30% easy).
- Ran PI-ResNet and CQT-DeiT on exactly the same `pair_id` values.
- Validated CQT preprocessing against stored 0222 PNGs before caching 15,000 event spectra.
- Froze unified SIS and PM prediction files and recorded SHA-256 sums.

## Core statistics

- Calibrated thresholds at FPP 1e-2, 1e-3, and 1e-4 on calibration background only.
- Evaluated achieved FPP and positive efficiency on the final evaluation partition.
- Completed 10,000 source-block bootstrap replicates.
- Completed exact McNemar, paired block-AUC, paired fixed-FPP efficiency, selection-function, and SNR/y matching analyses.
- Completed a 60-trial discrete peak-alignment lens-redshift sanity check.

## Archival release

- Built and published the v1.0 bundle: 159 payload files, 1,906,391,230 bytes, under DOI 10.5281/zenodo.21311078.
- Audited the CQT training pair provenance, found the split failure, rebuilt the pair set, retrained the baseline, and regenerated every derived result.
- Added twenty training instances, four extra seeds per architecture per lens family, for the instance-variability study.
- Decomposed the threshold-inclusive interval into calibration and evaluation components, and computed it per training instance rather than for the archived instance alone.
- Added generating code for the two analytic lensing figures, closing the last figure-provenance gap.
- Rebuilt the v2.1 bundle at tag apjs-resubmission-v2.1: 395 hash-verified payload files, 2,944,952,330 bytes, deterministic gzip archive of 2,654,423,871 bytes, SHA-256 fa659e633281032f64d3d7e04b5d67b2ec90ed229ef7b9f7624219013ab83c97. Deposition pending account authorization; the concept DOI 10.5281/zenodo.21311077 is the citable identifier.

## Additional completion analyses

- Completed SNR/y matched residual-gap bootstrap and weight/effective-sample-size diagnostics.
- Completed the 500-source E7 probe and stopped higher-mode/inclination subdivision after the pre-specified null first-stage result.
- Repeated cross-lens transfer in both directions for both architectures on the shared 0228 manifests with target-family calibration.
- Benchmarked model and preprocessing throughput, explicitly including CQT construction cost.
- Verified that logit-space calibration does not change the 1e-3 or 1e-4 selected pair sets.
- Formally downgraded historical LIGO results; no contaminated historical table is permitted as independent-test evidence.
