# ApJS revision: realistic populations

Scripts added for the revision requested by the ApJS reviewer (report of November 2026):
the uniform-prior simulations are replaced by an astrophysical population drawn with
`ler`, at O4 and at ET sensitivity, and the frozen workflow of
`experiments/reproducibility/` is rerun on it unchanged.

| Script | Purpose | Environment |
|---|---|---|
| `ler_population_rev.py` | LVK-like BBH population (GWTC-3 Power Law + Peak masses, star-formation merger rate, sources to z = 10) with EPL + shear galaxy lenses; rates; 6,000 detectable lensed systems and 10,000 detectable unlensed events per detector configuration | `ler` venv |
| `generate_strain_rev.py` | whitened single-detector 24 s segments in the repository's catalog layout; two disjoint catalogs (`0222` training, `0228` threshold-setting and measurement) from one population draw | `ler` venv (bilby, lalsuite) |
| `run_pipeline_rev.sh` | the frozen workflow steps with their data/output arguments pointed at a revision root; `--families SIS` because the galaxy-lens population is a single family | repository environment |
| `chain_rev.sh` | catalog generation, prep, both trainings in parallel, evaluation | both |

The family label `SIS` in the revision data roots holds the galaxy-lens population; the
`LABEL_NOTE.md` written into each root says so. Detection is defined at the population
level (ler: two images with network optimal S/N above 8); the stored channel is the
detector with the highest optimal S/N among those sharing the reference PSD, and the
`*_optimal_SNR_*.npy` arrays hold that single-detector value, as in the paper.
