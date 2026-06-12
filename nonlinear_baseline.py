#!/usr/bin/env python3
"""Isolate temporal accumulation from nonlinearity in the sequential detector.

The sequential paper compares the recurrent CUSUM against a LINEAR per-token
baseline (LogReg), so its speedup conflates a better (nonlinear) per-token score
with the temporal accumulation. delay_at_arl0() runs ANY per-token posterior
through the same threshold- or CUSUM-detection at a matched ARL0, so we can drop
in a nonlinear per-token model (HistGBM) and read off the 2x2 decomposition:
  nonlinearity = LogReg-threshold -> HistGBM-threshold
  accumulation = HistGBM-threshold -> HistGBM-CUSUM
  context      = HistGBM-CUSUM    -> ForwardGRU-CUSUM
"""
import numpy as np

from run_learned_cusum import (first_onset, prob_path, cusum_path,
                               cusum_reference_value, sweep, at_arl0)

_PROB_GRID = np.unique(np.concatenate([
    np.linspace(0.0, 0.999, 400),
    1 - np.logspace(-5, -1, 200),
]))
_CUSUM_GRID = np.linspace(0.0, 120.0, 601)


def delay_at_arl0(probs_list, labs_list, mode, target_arl0):
    """Operating point (delay/edd/recall/arl0) of a per-token posterior at a
    matched ARL0, detected either by thresholding ('threshold') or by an explicit
    CUSUM on the log-odds ('cusum'). Returns None if no operating point qualifies.
    """
    labs_list = [np.asarray(l, dtype=np.float64) for l in labs_list]
    onsets = [first_onset(l) for l in labs_list]

    if mode == "threshold":
        paths = [prob_path(p) for p in probs_list]
        grid = _PROB_GRID
    elif mode == "cusum":
        ref, _, _ = cusum_reference_value(probs_list, labs_list)
        paths = [cusum_path(p, ref) for p in probs_list]
        grid = _CUSUM_GRID
    else:
        raise ValueError(f"unknown mode: {mode!r} (use 'threshold' or 'cusum')")

    clean_paths = [paths[i] for i, o in enumerate(onsets) if o is None]
    hallu_paths = [paths[i] for i, o in enumerate(onsets) if o is not None]
    hallu_onsets = [o for o in onsets if o is not None]

    rows = sweep(clean_paths, hallu_paths, hallu_onsets, grid)
    return at_arl0(rows, target_arl0)
