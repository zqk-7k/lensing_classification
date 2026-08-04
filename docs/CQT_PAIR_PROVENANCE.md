# CQT--DeiT training pair provenance and split audit

**Status: the audit FAILS. The CQT--DeiT training pair images do not respect the
source-level train/validation split on the second member of each pair. PI-ResNet
does. The independently generated 0228 evaluation catalog is unaffected.**

Produced by `experiments/reproducibility/audit_cqt_training_pairs.py`. Raw output:
`results/audit/cqt_training_pair_audit.json` and
`results/audit/cqt_training_pair_manifest_{sis,pm}.csv.gz`.

## Why the audit was needed

`train_cqt_deit.py` does not build pairs from two events at run time. It consumes
precomputed pair images, `lensed/pos_XXXX.png` and `unlensed/neg_XXXX.png`, and parses
a single trailing integer from each filename as the source ID that decides
train/validation membership. The released split manifest
(`results/training/cqt_deit_*/split_manifest.csv`) therefore records only
`image_path, label, source_id, split, pair_type` -- one ID per image.

From the released products alone it was impossible to check who the *second* member of
each pair was, and therefore impossible to verify that no source or unlensed event
crossed the train/validation boundary through that second member.

## Method

Each pair image is a 224x224 render of two 112x224 constant-Q magnitude
representations concatenated along the frequency axis and written with
`plt.imsave(..., cmap="viridis")`. The rendering is deterministic, so both members can
be recovered forensically:

1. Invert the viridis colormap to the normalized scalar field of each half.
   `plt.imsave` normalizes over the concatenated matrix, so each recovered half is a
   shared affine rescaling of the underlying spectrum.
2. Recompute the transform for every candidate event in the 0222 catalog: both lensed
   images of all 2,500 sources and all 5,000 unlensed events.
3. Identify each half by Pearson correlation, which is invariant to that affine
   rescaling.

**The identification is unambiguous.** Across all 10,000 halves (2,500 positive and
2,500 negative images per lens family, two halves each) the worst matched correlation
is **0.99986**, and **zero** halves fall below the 0.99 acceptance threshold. Unrelated
chirp spectrograms correlate at 0.5--0.7, so there is no ambiguity between the matched
member and any alternative. The positive images serve as a built-in control.

## What the construction actually is

Confirmed for both families:

| Image | Left half (top) | Right half (bottom) |
|---|---|---|
| `pos_i.png` | image 1 of source `i` | image 2 of source `i` |
| `neg_i.png` | image 1 of source `i` | **hard**: image 2 of a *different* source, drawn from all 2,500 sources; **easy**: an unlensed event, drawn from all 5,000 events |

The filename index `i` therefore correctly identifies the **left** member of both
positive and negative images, and `pos_i` and `neg_i` always land in the same split.
The realized negative composition is 69.6% hard / 30.4% easy (SIS) and 70.7% / 29.3%
(PM), consistent with the documented 70/30 configuration.

The failure is that the right member was drawn from the **full** pools, ignoring the
split.

## Audit results

| Check | SIS | PM |
|---|---|---|
| Every half identified (r >= 0.99) | pass (worst 0.999861) | pass (worst 0.999859) |
| `pos_i` == (img1[i], img2[i]) | pass | pass |
| `neg_i` left member == img1[i] | pass | pass |
| Left members disjoint across split | pass (0 crossing) | pass (0 crossing) |
| **Lensed sources disjoint on both sides** | **FAIL (482 crossing)** | **FAIL (484 crossing)** |
| **Train/validation unlensed events disjoint** | **FAIL (22 crossing)** | **FAIL (14 crossing)** |
| Unlensed events used outside their declared pool | 100 train, 118 val | 116 train, 124 val |

Concretely, of the 500 validation sources, **225 (SIS) and 226 (PM)** have their
image 2 present inside a *training* pair image as the right member of a hard negative.
In the other direction, 257 (SIS) and 258 (PM) training sources appear as right members
inside validation images.

## PI-ResNet is not affected

`train_pi_resnet.py` restricts both pools before constructing the datasets:

```python
unl_train = IndexedArrayView(unl, unl_train_idx)
unl_val   = IndexedArrayView(unl, unl_val_idx)
assert set(train_idx).isdisjoint(val_idx)
assert set(unl_train_idx).isdisjoint(unl_val_idx)
train_ds = PairDataset(l1, l2, unl_train, train_idx, mode="train")
val_ds   = PairDataset(l1, l2, unl_val,   val_idx,   mode="val")
```

`PairDataset` draws hard negatives with `np.random.choice(self.indices)`, where
`self.indices` is the split-specific source pool, and easy negatives from the
split-restricted unlensed view. Both members of every PI-ResNet training and validation
pair therefore stay inside their own split.

## What is and is not compromised

**Not compromised.** The 0228 catalog is generated independently from the 0222 catalog,
with zero source-parameter overlap confirmed by
`experiments/reproducibility/manifests/0222_0228_independence_audit.json`. Calibration
thresholds and all reported efficiencies, achieved false-positive probabilities,
selection functions, transfer results, and diagnostics are computed on 0228 pair
manifests that both networks score identically. No training source appears anywhere in
the evaluation catalog. **The reported test-set results are measurements of the
archived checkpoints on genuinely held-out data and are not invalidated by this
finding.**

**Compromised.**

1. The CQT--DeiT validation AUCs recorded in `results/training/cqt_deit_*/summary.json`
   (0.980088 SIS, 0.965068 PM) are optimistically biased, because roughly 45% of the
   validation sources contributed a waveform to the training images.
2. Checkpoint selection for CQT--DeiT maximized that biased validation AUC, so the two
   pipelines did not select their archived checkpoints on equally clean signals.
3. The manuscript statement that, for the baseline, "the paired CQT images are grouped
   by source index so that images sharing a source waveform cannot straddle the split"
   is **false as written**. Grouping by the parsed index constrains the left member
   only.

**Likely direction of the bias.** A validation set that overlaps training inflates
validation AUC and rewards memorization, so selecting the maximum-validation-AUC
checkpoint plausibly yields a checkpoint that generalizes slightly *worse* on the clean
0228 catalog. If so, the archived baseline is if anything slightly understated and the
reported PI-ResNet advantage is conservative. This is an argument about the direction
of an uncontrolled effect, not a measurement, and it should not be presented as one.

## Options

**A. Rebuild and retrain the baseline.** Regenerate the 0222 CQT pair images with
split-respecting negative sampling, retrain both CQT--DeiT models, re-run 0228
inference, and recompute every table and figure that involves the baseline. PI-ResNet
is unaffected: its checkpoints, scores, and manifests are reusable unchanged. The two
baseline training runs took 2,342 s and 2,286 s of wall time on one RTX 5000 Ada, so
the compute cost is modest; regenerating the pair images and re-running the analysis
chain is the larger part of the effort.

**B. Disclose without retraining.** Publish this audit and the recovered manifests,
correct the false grouping statement, report the CQT--DeiT validation AUCs as
optimistically biased, and state the bias direction. The archived instances and all
0228 results stand as measurements.

Option A is the defensible choice if the manuscript continues to present the comparison
as controlled at the source level, which is currently one of its stated contributions.
Option B requires weakening that claim explicitly wherever it appears.

## Recovered manifests

`results/audit/cqt_training_pair_manifest_{sis,pm}.csv.gz` contain one row per training
pair image with the fields the release previously lacked:

```
pair_image_path, label, negative_type, declared_source_id,
left_event_id, left_source_id, left_kind,
right_event_id, right_source_id, right_kind,
unlensed_event_id, split, match_r_left, match_r_right
```

These are sufficient to rebuild the exact 0222 CQT training pair set, or to re-derive
any split statistic, without access to the rendered images.
