#!/usr/bin/env python3
"""Check that a manuscript directory carries the figures this repository produced.

Two rounds of external review caught the same failure twice: a figure script was
corrected, the figure was regenerated here, and the stale copy stayed in the submission
package, so the paper shipped a figure that disagreed with its own tables. Nothing in
the workflow made that visible. This does.

It compares every PDF in the manuscript's figure directory against
`results/figures/manuscript/SHA256SUMS`, and exits non-zero on any mismatch, so it can
be run before `latexmk` and block a build rather than be noticed by a reviewer.

    python experiments/reproducibility/verify_manuscript_figures.py --figures <dir>
    python experiments/reproducibility/verify_manuscript_figures.py --figures <dir> --sync

`--sync` copies the canonical files over the stale ones and re-checks. Outside the
repository layout -- in a submission package, say -- pass `--registry` (and
`--canonical-dir` if syncing) to point at a shipped copy of the registry.

Figures with no entry in the registry are reported separately rather than silently
passed: `before_whited.png` is a legacy illustrative raster with no generating script,
and is the only file expected in that category.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "results" / "figures" / "manuscript"
REGISTRY = CANONICAL / "SHA256SUMS"
EXPECTED_UNREGISTERED = {"before_whited.png"}
# figures produced by other stages keep their own registries
EXTRA_SOURCES = {"e7_score_shift.pdf": ROOT / "results" / "diagnostics" / "type_ii"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_registry() -> dict[str, str]:
    if not REGISTRY.is_file():
        sys.exit(f"missing registry: {REGISTRY}\nrun the figure scripts first")
    entries = {}
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, name = line.split(maxsplit=1)
        entries[name.strip().lstrip("*")] = sha
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", required=True, help="manuscript figure directory")
    parser.add_argument("--sync", action="store_true",
                        help="copy the canonical figures over any that differ")
    parser.add_argument("--registry", help="SHA256SUMS to check against; defaults to the "
                                           "released registry in this repository")
    parser.add_argument("--canonical-dir", help="directory holding the released figures; "
                                                "only needed for --sync")
    args = parser.parse_args()

    global CANONICAL, REGISTRY
    if args.canonical_dir:
        CANONICAL = Path(args.canonical_dir)
    if args.registry:
        REGISTRY = Path(args.registry)
    elif args.canonical_dir:
        REGISTRY = CANONICAL / "SHA256SUMS"

    figures = Path(args.figures)
    if not figures.is_dir():
        sys.exit(f"not a directory: {figures}")
    registry = load_registry()

    stale, missing, unregistered, ok = [], [], [], []
    for path in sorted(figures.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".png"}:
            continue
        source = EXTRA_SOURCES.get(path.name, CANONICAL)
        if path.name not in registry and source is CANONICAL:
            unregistered.append(path.name)
            continue
        canonical = source / path.name
        if source is not CANONICAL:
            if not canonical.is_file():
                missing.append(path.name)
            elif digest(path) == digest(canonical):
                ok.append(path.name)
            else:
                stale.append(path.name)
            continue
        if not canonical.is_file():
            missing.append(path.name)
        elif digest(path) == registry[path.name]:
            ok.append(path.name)
        else:
            stale.append(path.name)

    if stale and args.sync:
        for name in stale:
            shutil.copy2(EXTRA_SOURCES.get(name, CANONICAL) / name, figures / name)
        print(f"synced {len(stale)} stale figure(s): {', '.join(stale)}")
        stale = [n for n in stale
                 if digest(figures / n) != digest(EXTRA_SOURCES.get(n, CANONICAL) / n)]

    for name in ok:
        print(f"  ok        {name}")
    for name in unregistered:
        flag = "expected" if name in EXPECTED_UNREGISTERED else "UNEXPECTED"
        print(f"  {flag:9} no registry entry: {name}")
    for name in missing:
        print(f"  MISSING   canonical file absent: {name}")
    for name in stale:
        print(f"  STALE     {name}: manuscript copy differs from the released figure")

    unexpected = [n for n in unregistered if n not in EXPECTED_UNREGISTERED]
    bad = len(stale) + len(missing) + len(unexpected)
    print(f"\n{len(ok)} verified, {len(stale)} stale, {len(missing)} missing, "
          f"{len(unexpected)} unexpected")
    if bad:
        print("FAIL: the manuscript does not carry the figures this repository produced.")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
