#!/usr/bin/env python3
"""Final fair comparison after the adversarial review.

The ablation confirmed: the recall lives in the COVARIANCE (quad-only) term, not the
mean (linear-only 0.17) or a constant drift (0.01); E_clean[increment] is negative so
there is no end-of-document drift. What remains from the critics:
  - compare against the BEST scalar baselines, not the weakest (ForwardGRU-threshold and
    HistGBM, not just ForwardGRU-CUSUM);
  - report RECALL and DELAY-AMONG-DETECTED separately (censored EDD ~ 80*(1-recall) is
    not independent evidence);
  - show the full recall-vs-ARL0 picture, not one budget.

Detector under test: quad-only covariance CUSUM (0.5 x'(S0^{-1}-S1^{-1})x, midpoint
reference) -- the part the ablation isolated as the real signal.
"""
import json
from pathlib import Path

import numpy as np

from run_extended import load_and_enrich_all, assemble_features
from run_learned_cusum import (
    cusum_reference_value, cusum_path, prob_path, first_onset, sweep, at_arl0,
)

HERE = Path(__file__).resolve().parent
SHRINK = 0.1
GRID = np.linspace(0, 400, 2001)
PROB_GRID = np.unique(np.concatenate([np.linspace(0, 0.999, 600), 1 - np.logspace(-6, -1, 300)]))
TARGETS = [100, 200]


def shrink_cov(X):
    S = np.cov(X, rowvar=False)
    return (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S)) + 1e-6 * np.eye(S.shape[0])


def cusum(inc):
    S, p = 0.0, np.empty(len(inc))
    for t in range(len(inc)):
        S = max(0.0, S + inc[t]); p[t] = S
    return p


def report(name, paths, onsets, grid):
    clean = [paths[i] for i, o in enumerate(onsets) if o is None]
    hallu = [paths[i] for i, o in enumerate(onsets) if o is not None]
    h_on = [o for o in onsets if o is not None]
    rows = sweep(clean, hallu, h_on, grid)
    cells = {}
    for g in TARGETS:
        op = at_arl0(rows, g)
        cells[g] = op
    s = "   ".join(
        f"ARL0={g}: recall={cells[g]['recall']:.2f} delay-det={cells[g]['delay']:.1f}"
        for g in TARGETS if cells[g])
    print(f"  {name:24s} {s}", flush=True)
    return {str(g): (cells[g] if cells[g] else None) for g in TARGETS}


def main():
    fwd = json.load(open(HERE / "directional_probs_seed42.json"))["ForwardGRU"]
    hgb = json.load(open(HERE / "histgbm_probs.json"))["HistGBM"]
    ref_f, _, _ = cusum_reference_value(fwd["probs"], fwd["labs"])
    ref_h, _, _ = cusum_reference_value(hgb["probs"], hgb["labs"])

    data = load_and_enrich_all()
    te = [np.asarray(f, float) for f in
          assemble_features(data["te_base"], data["te_nli"], data["te_lm"], True, True)]
    tr = [np.asarray(f, float) for f in
          assemble_features(data["tr_base"], data["tr_nli"], data["tr_lm"], True, True)]
    Xtr = np.vstack(tr); ytr = np.concatenate([np.asarray(l) for l in data["tr_labs"]])
    te_labs = [np.asarray(l, float) for l in data["te_labs"]]
    onsets = [first_onset(l) for l in te_labs]
    assert len(te) == len(fwd["probs"]) == len(hgb["probs"]), "doc count mismatch"

    c, h = Xtr[ytr < 0.5], Xtr[ytr > 0.5]
    mu0, mu1 = c.mean(0), h.mean(0)
    S0i, S1i = np.linalg.inv(shrink_cov(c)), np.linalg.inv(shrink_cov(h))
    Q = S0i - S1i
    quad = lambda X: 0.5 * np.einsum("ij,jk,ik->i", X, Q, X)
    # midpoint reference for quad on TRAIN
    qc = np.concatenate([quad(d)[np.asarray(l) < 0.5] for d, l in zip(tr, data["tr_labs"])])
    qh = np.concatenate([quad(d)[np.asarray(l) > 0.5] for d, l in zip(tr, data["tr_labs"])])
    kq = 0.5 * (qc.mean() + qh.mean())

    quad_paths = [cusum(quad(X) - kq) for X in te]
    print("Fair comparison -- recall and delay-among-detected reported SEPARATELY:")
    out = {}
    out["ForwardGRU-threshold"] = report("ForwardGRU-threshold", [prob_path(p) for p in fwd["probs"]], onsets, PROB_GRID)
    out["ForwardGRU-CUSUM"] = report("ForwardGRU-CUSUM", [cusum_path(p, ref_f) for p in fwd["probs"]], onsets, GRID)
    out["HistGBM-threshold"] = report("HistGBM-threshold", [prob_path(p) for p in hgb["probs"]], onsets, PROB_GRID)
    out["HistGBM-CUSUM"] = report("HistGBM-CUSUM", [cusum_path(p, ref_h) for p in hgb["probs"]], onsets, GRID)
    out["covariance(quad)"] = report("covariance(quad-only)", quad_paths, onsets, GRID)

    print("\nHonest read: covariance wins ONLY if its recall exceeds the BEST scalar's recall "
          "(threshold/HistGBM), accepting it is slower (higher delay-among-detected).")
    json.dump(out, open("covariance_final.json", "w"), indent=2)
    print("Saved -> covariance_final.json")


if __name__ == "__main__":
    main()
