# Frozen experiment release v1

This release freezes the complete reproducibility payload for the calibrated 0222-training/0228-evaluation experiments.

Included large artifacts:

- Two PI-ResNet checkpoints: SIS noisy and PM noisy.
- Two CQT-DeiT checkpoints: SIS noisy and PM noisy.
- The pinned `deit_tiny_distilled_patch16_224-b40b3cf7.pth` initialization.
- Five float32 CQT event caches covering SIS image 1/2, PM image 1/2, and the shared unlensed pool.
- All source manifests, per-pair predictions, training records, derived statistics, reports, and figures selected by `build_zenodo_manifest.py`.

The CQT caches are derived intermediates for the 0228 holdout catalog. Raw 0222/0228 waveform catalogs are not included in this bundle.

Every manifest payload file is verified by size and SHA-256 before archive creation. The final archive has a separate SHA-256 sidecar and an `archive_info.json` record in `release/zenodo/dist/`.

## Measured release size, v2.1.1

- Manifest payload: 396 files, 2,944,958,395 bytes (2.74 GiB).
- Archive members: 404 files, including eight release-control files.
- Deterministic gzip archive: 2,654,425,064 bytes (2.47 GiB),
  SHA-256 `604ea06bd7263042418440a6a8eed9c8659d2991ec3c10b6be8ef21b618ffbf2`.
- Built at tag `apjs-resubmission-v2.1.1`.

The manuscript statement should use these measured values. Suggested wording:

> A frozen release containing the trained checkpoints, the pinned DeiT initialization
> weights, the CQT caches, the twenty additional training instances, and all derived
> results (396 manifest payload files; 2.94 GB uncompressed and 2.65 GB as a
> gzip-compressed archive; hash-verified against the committed manifest) is deposited on
> Zenodo under the concept DOI `10.5281/zenodo.21311077`, which always resolves to the
> latest version.

## What changed from v1.0

- The CQT--DeiT checkpoints are retrained on the rebuilt, split-respecting 0222 pair set
  (see `docs/CQT_PAIR_PROVENANCE.md`). The superseded checkpoints are recorded under
  `superseded` in the checkpoint registry and are not shipped.
- Sixteen additional checkpoints, four seeds per architecture per lens family, are
  included for the instance-variability study, together with their per-pair 0228
  predictions and training records.
- The provenance audit, the recovered manifests of the original pair set, and the
  rebuilt pair manifests are included.
- All derived results are regenerated: core analysis, post-hoc robustness, transfer,
  SNR matching, logit tail, and the manuscript figure set.

## Deposition status

Version v1.0 was published on 2026-07-11 under DOI `10.5281/zenodo.21311078`, licensed
CC BY 4.0. The concept DOI `10.5281/zenodo.21311077` always resolves to the latest
version and is the one the manuscript cites, so the text does not have to change when a
new version is deposited.

Version v2.1.1 is built and verified in `release/zenodo/dist/` but **not yet deposited**:
publishing requires an authenticated Zenodo account. To deposit it, open the existing
record, choose "New version", upload the four files from `dist/`, apply the metadata in
`metadata.example.json`, and publish. The record's related identifier for the GitHub
repository must also be corrected to `lensing_classification`; the v1.0 record still
carries the old misspelling.
