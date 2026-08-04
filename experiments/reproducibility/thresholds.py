"""The single operating-point threshold rule used across the analysis and figures.

The locked protocol defines the threshold at a target per-pair false-positive
probability as the k-th largest calibration-background score with
k = round(target * N), and retains a pair when its score is >= that threshold, so
that in the absence of ties exactly k background pairs lie at or above it.

`analyze_0228_core.py` realizes this as
`np.quantile(scores, 1 - target, interpolation="higher")`, which is identical for
the background sizes and operating points used here: `method="higher"` returns the
value at ascending index `ceil((1 - target) * (N - 1))`, which for N = 100,000 and
targets 1e-2, 1e-3, 1e-4 is exactly `N - k`. The default `np.quantile` interpolates
linearly between order statistics and is **not** equivalent; every script that
recomputes a threshold must use the rule below rather than the default.
"""

from __future__ import annotations

import numpy as np


def kth_threshold(scores, target: float) -> float:
    """k-th largest score, k = round(target * len(scores)), k >= 1."""
    scores = np.asarray(scores)
    k = max(1, int(round(target * len(scores))))
    return float(np.partition(scores, -k)[-k])
