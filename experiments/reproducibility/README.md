# Reproducibility workflow

This directory contains the executable workflow used to train, freeze, evaluate,
and diagnose the two ranking models. Dataset labels `0222` and `0228` are immutable
catalog version identifiers: `0222` supplies training/validation data and `0228`
supplies disjoint calibration/final-evaluation data.

## Stages

1. `audit_0222_0228.py` verifies catalog independence.
2. `build_split_0222.py` creates the shared source-level training split.
3. `train_pi_resnet.py` and `train_cqt_deit.py` train the four SIS/point-mass models.
4. `build_0228_pair_manifests.py` freezes calibration/evaluation pairs and source blocks.
5. `prepare_cqt_cache_0228.py` prepares CQT arrays.
6. `infer_pi_0228.py` and `infer_cqt_0228.py` evaluate identical pair manifests.
7. `analyze_0228_core.py` performs calibration, fixed-FPP evaluation, McNemar tests,
   paired AUC/efficiency comparisons, and source-block bootstrap.
8. `analyze_snr_matching_uncertainty.py` performs SIS--point-mass reweighting.
9. `make_core_figures.py` produces the locked analysis figures under `results/core/figures/`.
10. `analyze_cross_lens_transfer.py`, `run_e7_typeii_probe.py`, and
    `zl_invariance_sanity.py` implement secondary transfer and diagnostic analyses.

## Post-hoc diagnostics and manuscript figures

These run from the released per-pair scores alone and require no checkpoints,
catalogs, or GPU. They are secondary to the locked analysis and are labeled as
post hoc wherever they are reported.

- `posthoc_diagnostics.py` -> `results/core/posthoc_diagnostics.json`: detected-pair
  subset at the unchanged frozen thresholds, delete-one-block jackknife of the
  primary paired difference, and paired block accuracy with discordant counts.
- `analyze_posthoc_robustness.py` -> `results/core/posthoc_robustness.json`:
  threshold-inclusive block bootstrap that resamples the calibration blocks and
  re-derives the threshold in every replicate; fully detected-event analysis with
  the background filtered and the thresholds re-calibrated; and the amortized
  catalog-scale cost derived from `results/benchmarks/throughput/throughput.json`.
  The script asserts that its empirical-quantile rule reproduces the published
  thresholds exactly before bootstrapping.
- `make_figure1.py`, `make_figures_v22.py`, `regen_selection.py`, and
  `regen_snr_matched.py` -> `results/figures/manuscript/`: the manuscript figure
  set. These annotate the frozen values read from `core_results.json` rather than
  recomputing them, so the figures cannot drift from the primary tables. They
  write to a directory separate from `results/core/figures/` so that the locked
  analysis outputs and their `SHA256SUMS` registry are never overwritten.

Run every command from the repository root. Use `--help` for script-specific
arguments. Raw datasets default to `data/`; model weights and generated caches
default to `artifacts/`. Frozen compact outputs are published under `results/`.

The exact operating points, partitions, block construction, confidence intervals,
and primary outputs are defined in `docs/EVALUATION_PROTOCOL.md`. Do not change
these rules when reproducing the reported confirmatory analysis.
