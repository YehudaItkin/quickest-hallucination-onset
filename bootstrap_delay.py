#!/usr/bin/env python3
"""Fast bootstrap CIs for the decomposition delays.

Operating thresholds are fixed once on the full data (the ARL0=100 point); then we
resample the hallucination documents with replacement and recompute the
delay-among-detected at those fixed thresholds. This is a conditional bootstrap
(condition on the operating point) -- it captures the document-level variance of
the delay without re-matching ARL0 on every resample, so it runs in seconds.
Reports 95% CIs for the four cells and for the three paired drops.
"""
import json
from pathlib import Path

import numpy as np

from nonlinear_baseline import delay_at_arl0
from run_learned_cusum import (first_onset, prob_path, cusum_path,
                               cusum_reference_value, edd_on_hallu)

HERE = Path(__file__).resolve().parent
B = 1000
TARGET = 100


def main():
    dp = json.load(open(HERE / "directional_probs_seed42.json"))
    hg = json.load(open(HERE / "histgbm_probs.json"))
    labs = dp["ForwardGRU"]["labs"]
    cells = [("LogReg-threshold", "threshold", dp["LogReg"]["probs"]),
             ("HistGBM-threshold", "threshold", hg["HistGBM"]["probs"]),
             ("HistGBM-cusum", "cusum", hg["HistGBM"]["probs"]),
             ("ForwardGRU-cusum", "cusum", dp["ForwardGRU"]["probs"])]

    onsets = [first_onset(np.asarray(l, dtype=np.float64)) for l in labs]
    hallu_idx = [i for i, o in enumerate(onsets) if o is not None]
    hallu_onsets = [onsets[i] for i in hallu_idx]

    prepared = {}
    print("Full-data operating points (ARL0=100):")
    for nm, mode, probs in cells:
        op = delay_at_arl0(probs, labs, mode, TARGET)  # fix threshold on full data
        thr = op["threshold"]
        if mode == "threshold":
            paths = [prob_path(probs[i]) for i in hallu_idx]
        else:
            ref, _, _ = cusum_reference_value(probs, labs)
            paths = [cusum_path(probs[i], ref) for i in hallu_idx]
        prepared[nm] = (paths, thr)
        print(f"  {nm:<20}: delay={op['delay']:5.1f} recall={op['recall']:.2f} "
              f"arl0={op['arl0']:.0f} (thr={thr:.4g})")

    M = len(hallu_idx)
    rng = np.random.RandomState(0)
    order = [c[0] for c in cells]
    samples = {nm: np.empty(B) for nm in order}
    for b in range(B):
        idx = rng.randint(0, M, M)
        for nm in order:
            paths, thr = prepared[nm]
            hp = [paths[i] for i in idx]
            ho = [hallu_onsets[i] for i in idx]
            _, d, _ = edd_on_hallu(hp, ho, thr)
            samples[nm][b] = d

    def ci(a):
        a = a[~np.isnan(a)]
        return float(np.median(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    print("\nDelay-among-detected at ARL0=100, bootstrap median [95% CI]:")
    out = {}
    for nm in order:
        m, lo, hi = ci(samples[nm])
        out[nm] = [m, lo, hi]
        print(f"  {nm:<20}: {m:5.1f} [{lo:4.1f}, {hi:4.1f}]")

    print("\nDrops (paired bootstrap), median [95% CI]:")
    drops = {}
    for label, a, b_ in [("nonlinearity", "LogReg-threshold", "HistGBM-threshold"),
                         ("accumulation", "HistGBM-threshold", "HistGBM-cusum"),
                         ("context", "HistGBM-cusum", "ForwardGRU-cusum")]:
        d = samples[a] - samples[b_]
        d = d[~np.isnan(d)]
        m, lo, hi = float(np.median(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        sig = "significant" if lo > 0 else "n.s. (CI crosses 0)"
        drops[label] = [m, lo, hi]
        print(f"  {label:<13}: {m:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]  {sig}")

    json.dump({"B": B, "cells": out, "drops": drops}, open(HERE / "bootstrap_delay.json", "w"),
              indent=2, default=float)
    print("\nSaved -> bootstrap_delay.json")


if __name__ == "__main__":
    main()
