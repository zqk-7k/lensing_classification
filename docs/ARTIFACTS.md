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

The frozen bundle is archived under the concept DOI 10.5281/zenodo.21311077, which
always resolves to the latest version (CC BY 4.0). Version v2.2.1 carries 399 hash-verified
payload files totalling 2,944,973,191 bytes. The compressed archive's size and SHA-256
are recorded outside the payload -- in the `.sha256` sidecar, the `archive_info.json`
record, and the Zenodo file listing -- because a file inside the archive cannot state the
archive's own checksum without changing it. The payload contains:

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
