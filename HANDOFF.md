# Handoff — where this work stands

Written so that a fresh session, or a different person, can pick this up without the
conversation that produced it. Everything below is checkable from the repository.

## Canonical state

| | |
|---|---|
| Repository | `github.com/zqk-7k/lensing_classification` |
| Latest tag | `apjs-resubmission-v2.2.2` |
| Manuscript package | v2.5 rev2 |
| Zenodo concept DOI | `10.5281/zenodo.21311077` (always the latest version) |
| Zenodo published | v2.2.1, `10.5281/zenodo.21817506` |
| Zenodo pending | v2.2.2, draft `21820266`, upload incomplete |

The v2.2.2 tag is cut on a commit that already carries its own `MANIFEST.json`, so the
tag, the working tree and the archive built from it describe the same 399 payload files.
`release/zenodo/RELEASE_NOTES.md` explains why the compressed size and archive SHA-256
are kept outside the payload: a file inside an archive cannot state that archive's own
checksum without changing it, and earlier tags shipped stale documentation for exactly
that reason.

## Scientific position

- **SIS at the primary operating point** is the established result: +20.8 percentage
  points, separated from zero under both the locked and the threshold-inclusive
  interval, and positive in all five training instances.
- **PM is directional, not established.** The locked interval excludes zero but the
  threshold-inclusive one does not, and four of five instances resolve a positive
  difference while the archived one does not.
- **At 1e-4** neither the locked configuration nor the five-instance ensemble resolves a
  difference for either family; some individual instances do.
- The final CQT--DeiT baseline is a **protocol-preserving corrective reanalysis**, not a
  blind evaluation. PI-ResNet retains its original blind status.

`docs/RESULTS.md` and the manuscript's `CHANGES_v2.5.md` carry the full picture.

## What is NOT in this repository

**The manuscript source.** `main.tex`, the two letters, `references.bib` and the figure
copies live only in the delivered package `PI-ResNet_ApJS_v2.5_rev2.tar.gz`. A new
session needs that archive uploaded to it before it can continue editing the paper.
Whether the manuscript should live in this repository is an open decision for the
authors; it is currently kept out because the repository is public.

Everything the manuscript *cites* — every number, script, figure and result — is here.

## Work in flight

**Zenodo v2.2.2 upload.** Draft `21820266` holds two of four files. The 2.65 GB archive
failed on a transient HTTP 502 and the link has been slow since. The deposition script
is resumable; already-uploaded files are skipped by checksum:

    export ZENODO_TOKEN=...
    python release/zenodo/deposit_new_version.py --draft 21820266 --attempts 8

It stops at the draft. Publishing stays a human decision.

## Open items, authors only

1. **Publish the v2.2.2 draft** once its upload completes. It supersedes the published
   v2.2.1, which predates the bin-edge correction. A published Zenodo version cannot be
   altered, so superseding is the only route and the version chain is the record. The
   manuscript cites the concept DOI, so no text changes follow.
2. **Confirm the generative-AI disclosure.** The manuscript names Anthropic's Claude and
   describes the use as language editing, code review and figure-script drafting. Needs
   author confirmation of: the product (Claude Code, Claude Chat, or both), the model
   version, the date range, and whether that description covers the real scope — the
   provenance audit, retraining orchestration, analysis scripts and manuscript edits of
   the later rounds go beyond figure-script drafting.
3. **Revoke the Zenodo API tokens** used during this work. They appeared in plain text in
   the working conversation.

## Work requested but not finished

**Plain-language pass over the manuscript.** The authors asked that jargon be reduced so
that an astrophysicist reads it without friction, giving "operating point" as an example.
A scan of `main.tex` found the main candidates and their counts:

| Term | Count | Note |
|---|---|---|
| operating point | 47 | field-standard in GW searches, but dense at this frequency |
| efficiency | 62 | standard, keep |
| post-hoc | 13 | could be "after the fact" in most places |
| source-block | 13 | standard for the bootstrap, keep |
| threshold-inclusive | 9 | invented compound; "which also resamples the calibration set" |
| ranking statistic | 9 | the referee asked for this term; keep |
| unblind | 8 | standard in blind analysis; keep |
| false-positive probability | 6 | keep |
| prior-marginalized | 5 | heavy; "averaged over the other parameters" |
| selection-efficiency projection | 4 | heavy |
| pairwise-interaction | 4 | fine |
| discordant | 3 | statistical jargon |
| protocol-preserving corrective reanalysis | 2 | precise but heavy |
| one-factor-at-a-time | 1 | heavy |
| conditioning-induced | 1 | heavy |
| instance-to-instance | 1 | "run-to-run" is plainer |

Judgement needed before editing: several of these came from the referee's own request
(M2 asked for the ranking-statistic framing and fixed-false-positive reporting), so
replacing them wholesale would move the paper away from what the referee asked for. The
sensible split is to simplify invented compounds and heavy nominalizations while keeping
the vocabulary the referee and the GW-search literature actually use.

## How to resume

1. `git clone` this repository and read this file, then `CHANGES_v2.5.md` in the
   manuscript package for the full revision history.
2. Upload `PI-ResNet_ApJS_v2.5_rev2.tar.gz` if the paper text needs editing.
3. The analysis server holds the catalogs, checkpoints and CQT caches; the repository
   holds everything needed to reproduce the reported statistics without it.
4. Before compiling the manuscript, run the figure gate:

       python scripts/verify_manuscript_figures.py --figures figures --registry figures/SHA256SUMS

   It fails if the package carries any figure other than the released one. Two rounds of
   review caught exactly that error before the gate existed.
