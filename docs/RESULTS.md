# Independent evaluation results

## Status

The clean 0222 training protocol, independent 0228 calibration/evaluation protocol, four-model inference, core fixed-FPP analysis, selection functions, SNR/y reweighting, lens-redshift sanity check, and minimal E7 type-II diagnostic are complete. The frozen Zenodo deposition is archived under the concept DOI 10.5281/zenodo.21311077 (v2.0 built and verified; deposition pending account authorization).

## Data roles

- Catalog 0222: training and checkpoint-selection validation only.
- Catalog 0228 calibration partition: operating-threshold and bin-edge calibration only.
- Catalog 0228 evaluation partition: final metrics, paired comparisons, and selection functions only.
- Catalog 0228 is an independently generated IID holdout from the same simulation priors, not an OOD or external dataset.

## Frozen models

The frozen release contains PI-ResNet and CQT-DeiT (SEMD-inspired) checkpoints for ET SIS-Noisy and PM-Noisy. Exact paths, best epochs, validation AUC values, and SHA-256 sums are recorded in the checkpoint registry.

## Primary independent-test results

PI-ResNet evaluation AUC is 0.98877 for SIS and 0.98522 for PM, compared with 0.97670 and 0.96124 for CQT-DeiT. The paired block-AUC differences are 0.01207 (95% CI 0.00948--0.01421) and 0.02398 (0.02050--0.02713).

At the primary target FPP of 1e-3, PI-ResNet efficiency is 0.5354 (0.5097--0.5629) for SIS and 0.2274 (0.2109--0.2457) for PM. CQT-DeiT reaches 0.3274 (0.3011--0.3514) and 0.1726 (0.1611--0.1841). The paired PI-minus-CQT efficiency gains are 0.2080 (0.1771--0.2394) and 0.0549 (0.0311--0.0794).

These baseline values follow the retraining on the corrected, split-respecting 0222 pair set (see `docs/CQT_PAIR_PROVENANCE.md`). PI-ResNet is unchanged. The threshold-inclusive intervals separate the two families: SIS is [+0.103,+0.287] with 3e-4 of replicates at or below zero, PM is [-0.080,+0.180] with 26 percent, so the PM advantage at the primary operating point is not established once calibration error is propagated.

The ordering is not uniform at target FPP 1e-4. For SIS, PI-ResNet is lower by 0.0140 (CI -0.0371 to 0.0131). For PM it is lower by 0.0103 (CI -0.0200 to 0.0043). Both intervals include zero, so the two statistics are indistinguishable in the extreme tail. Accordingly, 1e-3 is the preregistered primary operating point and 1e-4 is interpreted as a tail diagnostic.

## Selection effects

Scores and efficiencies depend strongly on weaker-image SNR. Efficiency decreases with increasing impact parameter and magnification imbalance. In the present lens models, impact parameter and flux ratio are tightly linked and must not be presented as independent discoveries.

## SNR/y matching

Common-support reweighting reduces, but does not remove, the SIS-PM efficiency gap. For PI-ResNet the common-support gap changes from about 0.310 before weighting to 0.220 after weighting. The matched residual gap is 0.2201 with 95% CI [0.1922, 0.2489]. For CQT-DeiT it is 0.1122 [0.0787, 0.1462], half the PI-ResNet value, so the residual is not a pipeline-independent quantity. Effective sample sizes are approximately 1,156 for SIS and 1,468 for PM; maximum weights are 7.35 and 4.59, respectively. SNR/y therefore explains a substantial fraction, not all, of the lens-family gap.

## E7 controlled type-II diagnostic

The pre-specified 500-source physical/no-Morse comparison is a null result. SIS has mean physical-minus-control score -0.0137 (CI -0.0456 to 0.0180; Wilcoxon p=0.453), while PM has -0.00384 (-0.0284 to 0.0214; p=0.848). Fixed-1e-3-threshold efficiency differences also include zero. The present network therefore shows no measurable sensitivity to the controlled intervention, and no higher-mode/inclination subdivision is pursued.

## Cross-lens transfer

Transfer is strongly asymmetric. On PM evaluation pairs both architectures improve under the SIS-trained checkpoint: 0.227 to 0.398 for PI-ResNet (+0.170) and 0.173 to 0.281 for CQT-DeiT (+0.108). On SIS evaluation pairs the PM-trained PI-ResNet loses more than half its efficiency (0.535 to 0.221, -0.315) while the baseline is barely affected (0.327 to 0.298, -0.029). The direction shared by both pipelines is therefore the PM one: training on the brighter SIS positives yields a better PM rule than training on PM itself.

## Throughput

On an NVIDIA RTX 5000 Ada Generation with batch size 256, PI-ResNet GPU inference takes 0.728 ms per pair and CQT-DeiT model inference takes 0.218 ms per pair. The 2D model is therefore the faster of the two per forward pass; the cost difference resides entirely in preprocessing. Direct PI preprocessing costs 0.514 ms per pair, whereas serial CQT construction costs 66.0 ms per pair (both figures cover the two segments of a pair). A 16-thread CQT trial was slower because of library-level thread oversubscription and is excluded from the primary comparison.

Two regimes must be distinguished, and the headline ratio applies to only one of them.

- Cold start, transform recomputed for every pair: end-to-end costs are 1.24 ms per pair for PI-ResNet (about 800 pairs/s) and 66.3 ms for CQT-DeiT, a roughly 53-fold advantage for the direct time-domain pipeline.
- Amortized over a catalog: in a screen of N events each event enters N-1 pairs, so both pipelines can transform each event once and cache it. Per-event preprocessing costs 0.257 ms (time domain) and 33.019 ms (CQT), and the per-pair cost tends to the forward pass alone. With `c_pair = c_gpu + 2*c_event/(N-1)`, the cached per-pair costs are 0.785 vs 7.555 ms at N=10, 0.733 vs 0.885 ms at N=100, and 0.728 vs 0.284 ms at N=1000.

The ordering therefore crosses over at N = 129.4 events, beyond which the 2D pipeline is the cheaper of the two per pair. The 53-fold figure must not be quoted without stating that it describes a cold-start, no-cache implementation. At catalog scale the two pipelines are comparable in cost, and the case for the time-domain route rests on its calibrated efficiency rather than on speed. Both remain far below the minutes-to-days per pair of Bayesian analyses.

## Logit tail check

Recalibration in pre-sigmoid/logit space selects exactly the same evaluation pairs as probability-space calibration at target FPP 1e-3 and 1e-4 for both models and both lens families (Jaccard 1.0). A monotonic logit transform therefore does not remove the SIS 1e-4 reversal. Probability saturation may be described as a display/numerical-conditioning limitation, but it is not an empirically established mechanism for the reversal in these stored predictions.

## Lens-redshift sanity check

Across 60 deterministic delay interventions, the maximum classifier-input difference after peak realignment is 2.38e-7 and the maximum relative L2 difference is 2.14e-8. This establishes numerical invariance only within the present discrete peak-aligned verification setup at fixed source and y.

## Supported conclusion

The evidence supports PI-ResNet as a calibrated time-domain pair-ranking statistic with materially higher efficiency than the CQT-DeiT baseline at the primary 1e-3 per-pair FPP for SIS, and at 1e-2 for both families. It does not support the PM advantage at 1e-3 once calibration error is propagated, any difference at 1e-4, real-noise robustness, catalog-level FAR claims, complete SNR explanation of the SIS-PM gap, or measurable Morse-phase sensitivity in the minimal E7 probe.

## Training-instance variability

Five instances per architecture and family, varying only the seed. On SIS the paired 1e-3 difference spans +20.8 to +27.6 pp (mean +22.9, sd 2.7); on PM it spans +5.5 to +25.9 pp (mean +18.5, sd 8.9). The archived seed-42 instance is the smallest of the five for both families, so the locked analysis is conservative. Instance reproducibility degrades sharply with the lower effective SNR of the PM population. Score-averaged five-instance ensembles reach +15.4 pp [+11.9,+19.4] on SIS and +18.0 pp [+11.1,+23.2] on PM at 1e-3, both excluding zero; at 1e-4 the ensembles are indistinguishable on both families. See `results/seeds/`.

## Reproducibility

All compact code, manifests, per-pair predictions, statistical tables, and hashes are stored in GitHub. Large checkpoints and CQT caches are hash-registered in `release/zenodo/MANIFEST.json` and published in the Zenodo release under the concept DOI 10.5281/zenodo.21311077.
