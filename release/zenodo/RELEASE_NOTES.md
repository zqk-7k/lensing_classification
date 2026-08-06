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

## Where the numbers live, and why they are split

A file inside an archive cannot state that archive's own SHA-256: writing the value in
changes the bytes and invalidates it. Several earlier tags shipped documentation
describing a bundle other than the one built from them for exactly this reason. The two
kinds of fact are therefore kept apart:

*Inside the tagged payload* (README.md, docs/ARTIFACTS.md, MANIFEST.json): the version,
the payload file count, the payload byte total, and the per-file checksums.

*Outside it* (this file, the `.sha256` sidecar, `archive_info.json`, the Zenodo file
listing): the compressed archive size and the archive SHA-256.

## Measured release size, v2.2.2

- Manifest payload: 399 files. The byte total is in `MANIFEST.json`; it is not repeated
  here, nor in any file inside the payload, so that it has exactly one authority.
- Built at tag `apjs-resubmission-v2.2.2`, whose committed `MANIFEST.json` describes the
  same payload, so the tag and the archive agree by construction and this is checked.
- The compressed size and SHA-256 are written by `build_frozen_release.py` into
  `dist/*.archive_info.json` and `dist/*.tar.gz.sha256` at build time. They are not
  repeated here, so that regenerating the archive does not require editing a file that
  is inside it.

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

Version v2.2.2 is built and verified in `release/zenodo/dist/`:
publishing requires an authenticated Zenodo account. To deposit it, open the existing
record, choose "New version", upload the four files from `dist/`, apply the metadata in
`metadata.example.json`, and publish. The record's related identifier for the GitHub
repository must also be corrected to `lensing_classification`; the v1.0 record still
carries the old misspelling.
