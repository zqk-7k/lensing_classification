# Zenodo frozen release bundle

This directory defines and builds the frozen reproducibility archive.

## Build on the experiment server

```bash
python release/zenodo/prepare_zenodo_artifacts.py
python release/zenodo/build_zenodo_manifest.py
python release/zenodo/build_frozen_release.py
```

The first command maps the four final-v1 checkpoints, pinned DeiT initialization, and five CQT arrays into canonical `artifacts/` paths. It uses hard links when possible and verifies each source against the checkpoint registry or CQT metadata.

The second command hashes the complete payload and updates `MANIFEST.json` and `MANIFEST.sha256`. Commit those two files before describing the archive as hash-verified against the committed manifest.

The final command verifies every manifest row again and writes a deterministic archive, archive checksum, file list, and archive information record under `release/zenodo/dist/`. The `dist/` directory is intentionally excluded from Git.

## Published deposition

Version v1.0 is published under DOI 10.5281/zenodo.21311078. Version v2.0 is built and
verified in `dist/` but not yet deposited. Cite the concept DOI
[10.5281/zenodo.21311077](https://doi.org/10.5281/zenodo.21311077), which always
resolves to the latest version. Author names, ORCID identifiers, affiliations, and the
related repository identifier are in `metadata.example.json`.

To cut a new version, rebuild with the three commands above, confirm that
`MANIFEST.json` and `MANIFEST.sha256` are committed, and deposit as a new version of
the existing Zenodo record so that the concept DOI continues to resolve to the
latest release. Do not publish or cite a DOI while `metadata.example.json` contains
`REQUIRED_*` placeholders.
