# Artifact storage policy

All source code, locked manifests, audit reports, compact training records, and per-pair 0228 prediction scores are versioned in GitHub. Large binary artifacts are intentionally excluded from ordinary Git history and are identified by cryptographic hashes in the committed registries and metadata.

## Stored in GitHub

- Experiment and analysis source code.
- Evaluation protocol locks and Git tags.
- 0222 source split and 0228 shared pair manifests.
- Independence audits.
- Model configs, histories, split manifests, summaries, environment records, and train/validation predictions.
- PI-ResNet and CQT-DeiT per-pair 0228 prediction scores and metadata.
- CQT preprocessing validation and cache metadata.
- Statistical tables, figures, reports, and result hashes after completion.

## Published in the Zenodo release

The frozen bundle is published under DOI 10.5281/zenodo.21311078 (v1.0, CC BY 4.0):
159 hash-verified payload files, 1,906,391,230 bytes uncompressed, distributed as a
deterministic gzip archive of approximately 1.69 GB with a SHA-256 sidecar and an
`archive_info.json` record. It contains:

- PI-ResNet checkpoints (about 131 MB each; above GitHub's ordinary per-file limit).
- CQT-DeiT checkpoints.
- Official DeiT initialization weights.
- CQT event caches (250--500 MB per array).

The frozen checkpoint identities are recorded in `experiments/reproducibility/manifests/checkpoint_registry.json`. CQT cache hashes and shapes are recorded in `results/preprocessing/cqt_cache_metadata.json`. The Zenodo bundle was verified against these hashes before release, and can be re-verified after download against `release/zenodo/MANIFEST.json` and `release/zenodo/MANIFEST.sha256`.

## Not distributed

The raw 0222 and 0228 strain catalogs (hundreds of GB) are not included in either
GitHub or the Zenodo bundle. Their shapes, generation configurations, and checksums
are documented in `docs/DATASETS.md`, the generating programs are in
`src/generation/`, and the catalogs are available from the corresponding author on
reasonable request. The CQT training images derived from the 0222 catalog are
likewise not distributed; the parameters needed to regenerate them are recorded in
`docs/BASELINE_CONFIG.md`.
