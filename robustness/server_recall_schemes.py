#!/usr/bin/env python3
"""Direction 2: attack recall, not delay.

The censored EDD of 56-66 tokens is set by misses, not by detection speed: at
ARL0=100 every realistic detector catches only ~30% of onsets. Two angles on recall,
both on the ForwardGRU-CUSUM statistic, no GPU:

  recall vs ARL0 : the standard detector at looser false-alarm budgets. How much
                   recall does a tighter budget cost?
  persistence-k  : alarm only when the CUSUM stays above threshold for k consecutive
                   tokens (a confirmation/debounce step). A single noise spike no
                   longer fires, so at a fixed ARL0 the threshold can be LOWER, which
                   should raise recall at a small delay cost. Implemented as a standard
                   threshold on the running k-window minimum of the CUSUM path:
                   min_{t-k<i<=t} S_i >= h  iff  S stayed above h for k steps.
"""
import json
from pathlib import Path

import numpy as np

from run_learned_cusum import (
    cusum_reference_value, cusum_path, first_onset, sweep, at_arl0,
)

HERE = Path(__file__).resolve().parent
PROBS = HERE / "directional_probs_seed42.json"


def running_min(path, k):
    """Trailing k-window minimum; threshold on this = k-consecutive persistence."""
    if k <= 1:
        return path
    out = np.empty(len(path))
    for t in range(len(path)):
        out[t] = path[max(0, t - k + 1): t + 1].min()
    return out


def eval_at(paths, onsets, targets):
    clean = [paths[i] for i, o in enumerate(onsets) if o is None]
    hallu = [paths[i] for i, o in enumerate(onsets) if o is not None]
    h_on = [o for o in onsets if o is not None]
    rows = sweep(clean, hallu, h_on, np.linspace(0, 120, 801))
    res = {}
    for g in targets:
        op = at_arl0(rows, g)
        res[g] = op
    return res


def main():
    data = json.load(open(PROBS))
    fwd = data["ForwardGRU"]
    labs = [np.asarray(l, float) for l in fwd["labs"]]
    onsets = [first_onset(l) for l in labs]
    ref, _, _ = cusum_reference_value(fwd["probs"], fwd["labs"])
    base = [cusum_path(p, ref) for p in fwd["probs"]]

    print("Recall vs false-alarm budget (standard CUSUM):")
    res = eval_at(base, onsets, [10, 20, 50, 100, 200])
    out = {"recall_vs_arl0": {}, "persistence": {}}
    for g, op in res.items():
        if op:
            print(f"  ARL0={g:<4}: recall={op['recall']:.2f}  censored EDD={op['delay_edd']:.1f}  "
                  f"delay={op['delay']:.1f}  @ARL0={op['arl0']:.0f}")
            out["recall_vs_arl0"][str(g)] = {"recall": op["recall"], "edd": op["delay_edd"],
                                             "delay": op["delay"], "arl0": op["arl0"]}

    print("\nPersistence-k confirmation at ARL0=100 (debounce single spikes):")
    for k in [1, 2, 3, 5]:
        paths = [running_min(p, k) for p in base]
        op = eval_at(paths, onsets, [100])[100]
        if op:
            print(f"  k={k}: recall={op['recall']:.2f}  censored EDD={op['delay_edd']:.1f}  "
                  f"delay={op['delay']:.1f}  @ARL0={op['arl0']:.0f}")
            out["persistence"][str(k)] = {"recall": op["recall"], "edd": op["delay_edd"],
                                          "delay": op["delay"], "arl0": op["arl0"]}
    print("\nReading: persistence raises recall at fixed ARL0 if single-spike false "
          "alarms dominate; otherwise it only adds delay.")
    json.dump(out, open(HERE / "recall_schemes.json", "w"), indent=2)
    print(f"Saved -> {HERE/'recall_schemes.json'}")


if __name__ == "__main__":
    main()
