# CQT--DeiT (SEMD-inspired) baseline: complete configuration

This document collects, in one place, every setting needed to reproduce the
two-dimensional comparison baseline. It exists because the baseline configuration
was previously recoverable only by reading `experiments/reproducibility/train_cqt_deit.py`
together with `results/training/cqt_deit_*/config.json`, which made the baseline
look less fully specified than PI-ResNet. Nothing here is new: every value is the
one recorded in the frozen training runs, and the authoritative sources are cited
per section.

## Input representation

The baseline consumes precomputed pair images: the constant-Q magnitude
representations of the two candidate segments are composited into a single PNG,
and the network performs binary classification on that image. One image file
therefore corresponds to one candidate pair, in contrast to PI-ResNet's two-branch
Siamese input.

Constant-Q transform parameters (source: `experiments/reproducibility/prepare_cqt_cache_0228.py`):

| Parameter | Value |
|---|---|
| Library | `librosa.cqt` |
| Sampling rate `sr` | 2048 Hz (the decimated network rate) |
| `fmin` | 20.0 Hz |
| `n_bins` | 112 |
| `bins_per_octave` | 24 |
| `hop_length` | 16 |
| Cached array shape per event set | `(2500, 112, 224)`, float32 |

These are the parameters used to build both the 0228 evaluation caches and the
0222 training images. The equivalence was verified rather than assumed: the
reimplemented transform was compared pixel-by-pixel against the stored 0222 PNGs
before any caching, giving a maximum absolute pixel difference of 3 (out of 255)
and an exact-pixel fraction of 0.99943 (SIS) and 0.99900 (PM). The full check is in
`results/preprocessing/cqt_validation.json`, and the per-array shapes and SHA-256
sums of the five caches are in `results/preprocessing/cqt_cache_metadata.json`.

The 0222 training images themselves (`dataset_images_{SIS,PM}_noisy_cqt/`) are
derived intermediates and are not distributed; they are regenerable from the raw
strain catalogs with the parameters above. The 0228 caches are distributed in the
Zenodo bundle.

## Architecture

Source: `src/cqt_deit/model.py`.

- Backbone: `timm` model `deit_tiny_distilled_patch16_224`, embedding dimension 192.
- Initialization: the official distilled DeiT-tiny weights, pinned by file and hash
  (`deit_tiny_distilled_patch16_224-b40b3cf7.pth`, SHA-256
  `b40b3cf7d94a9cf8c7902e696de6aada1203d6eef58fe70c231f443426f845b3`). Training
  aborts if the file is absent; loading is strict apart from the two replaced heads.
- Backbone fine-tuning: full, `freeze_backbone=False`.
- Heads: both `head` and `head_dist` (the distillation token head) are replaced by
  the same structure — `LayerNorm` -> `FeatureGate(reduction=4)` -> `Dropout(0.5)`
  -> `Linear(192, 512, bias=False)` -> `BatchNorm1d(512)` -> `GELU` -> `Dropout(0.5)`
  -> `Linear(512, 2)`.
- `FeatureGate` is a squeeze-and-excitation-style channel gate:
  `Linear(C, C/4) -> ReLU -> Linear(C/4, C) -> Sigmoid`, applied multiplicatively.
- Output: 2-class logits; the reported score is `softmax(logits)[:, 1]`.

## Image transforms

Source: `src/cqt_deit/dataset.py`.

Training:
`Resize(224, 224)` -> `RandomResizedCrop(224, scale=(0.8, 1.0))` ->
`RandomHorizontalFlip(p=0.5)` -> `RandomRotation(10)` -> `ToTensor` ->
`Normalize(ImageNet mean/std)` -> `RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3))`.

Evaluation and inference:
`Resize(224, 224)` -> `ToTensor` -> `Normalize(ImageNet mean/std)`.

ImageNet statistics are mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
Images are converted to RGB on load.

## Optimization

Sources: `experiments/reproducibility/train_cqt_deit.py`, and the recorded values in
`results/training/cqt_deit_{sis,pm}_noisy_seed42/config.json`.

| Setting | CQT--DeiT | PI-ResNet (for comparison) |
|---|---|---|
| Epochs | 300 | 300 |
| Batch size | 32 | 64 |
| Optimizer | AdamW | AdamW |
| Initial learning rate | 5e-5 | 1e-4 |
| Weight decay | 5e-4 | 1e-4 |
| LR schedule | Cosine annealing, `T_max=300`, `eta_min=1e-6` | Cosine annealing over 300 epochs |
| Loss | Cross-entropy, `label_smoothing=0.1` | Binary cross-entropy with logits |
| Seed | 42 | 42 |
| Dataloader workers | 4 | 4 |
| Device | `cuda:0` | `cuda:0` |
| Checkpoint selection | Highest validation AUC | Highest validation AUC |

Determinism: `random`, `numpy`, and `torch` seeds are set, `cudnn.deterministic=True`
and `cudnn.benchmark=False`, and the training `DataLoader` uses a seeded generator.

The two architectures deliberately retain architecture-specific optimizers, batch
sizes, and augmentations, as is standard when comparing a convolutional network with
a fine-tuned image transformer. What is held identical is the source-level split, the
epoch budget, and the archived evaluation pair manifests.

## Split handling

The baseline reads the same shared split as PI-ResNet,
`experiments/reproducibility/manifests/split_0222_seed42.npz` (SHA-256
`f871af69d4aac09b9d05488b0370bbbfe001a73229df575b5b96cd97f8b37f52`). Image files are
grouped by the source index parsed from the filename, so images sharing a source
waveform cannot straddle the train/validation boundary. The training script asserts
that the train and validation source-ID sets are disjoint, and raises if any image
carries a source ID absent from the shared split. The realized assignment is written
to `results/training/cqt_deit_*/split_manifest.csv`.

## Hyperparameter search

The CQT resolution, analyzed frequency range, and learning rate were selected on the
0222 validation split before the evaluation protocol was locked; only the selected
configuration was carried into the frozen v1 training runs recorded here. The search
runs themselves predate the locked protocol and are not part of the frozen release.

## Frozen run outcomes

| Run | Best validation AUC | Runtime (s) | Checkpoint SHA-256 |
|---|---|---|---|
| `cqt_deit_sis_noisy_seed42` | 0.980088 | 2341.6 | `5c3d775e2991a21ac3331f93986bde9b2e9fe1d6824311c8a312e5f82e847952` |
| `cqt_deit_pm_noisy_seed42` | 0.965068 | 2285.9 | `82f3d394542d84be10e49392dbe86f20618e81cfe6bd16b045be290ea30da270` |

Both runs were executed at commit `4c168578d4e027a9a3b8d3482abd324bba13fdeb` under
Python 3.12.12, PyTorch 2.5.1+cu124, CUDA 12.4, on an NVIDIA RTX 5000 Ada Generation.
The checkpoints are distributed in the Zenodo bundle
(DOI 10.5281/zenodo.21311078) and registered in
`experiments/reproducibility/manifests/checkpoint_registry.json`.

## Reproduction

```bash
python experiments/reproducibility/train_cqt_deit.py --lens SIS
python experiments/reproducibility/train_cqt_deit.py --lens PM
```

All settings above are argparse defaults; pass `--data-root`/`--image-root` or set
`GW_DATA_ROOT` to point at the catalogs. Inference on the frozen 0228 manifests is
`prepare_cqt_cache_0228.py` followed by `infer_cqt_0228.py`.
