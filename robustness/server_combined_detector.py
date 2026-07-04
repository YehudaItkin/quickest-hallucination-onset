#!/usr/bin/env python3
"""Combine the two complementary detectors found in direction 1.

The learned ForwardGRU scalar is fast but low-recall (delay 11, recall 0.24 at
ARL0=100): it reads a mean-shift score. The full-covariance Gaussian LLR is slower but
high-recall (delay 17, recall 0.43): it reads the covariance reshaping the scalar
misses. They look complementary -- one a first-order (location) statistic, the other
second-order (shape). Combining them should keep the recall and recover the speed.

Three combined rules at matched ARL0, each token's two increments standardized to unit
clean-stream scale first:
  sum        : a single CUSUM on inc_logit_z + inc_gauss_z.
  max-channel: two CUSUMs, alarm on max(S_logit, S_gauss) >= h (MEI multichannel).
Reported against each channel alone and the references. Needs the saved ForwardGRU
posteriors AND the 33-d features for the SAME test documents (verified by label match).
"""
import json
from pathlib import Path

import numpy as np

from run_extended import load_and_enrich_all, assemble_features
from run_learned_cusum import cusum_reference_value, first_onset, sweep, at_arl0

HERE = Path(__file__).resolve().parent
PROBS = HERE / "directional_probs_seed42.json"
GAMMA_TARGETS = [50, 100, 200]
ARL0_MAIN = 100
SHRINK = 0.1


def shrink_cov(X):
    S = np.cov(X, rowvar=False)
    return (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S)) + 1e-6 * np.eye(S.shape[0])


def cusum(inc):
    S, path = 0.0, np.empty(len(inc))
    for t in range(len(inc)):
        S = max(0.0, S + inc[t])
        path[t] = S
    return path


def evaluate(name, paths, onsets, grid, targets=GAMMA_TARGETS):
    clean = [paths[i] for i, o in enumerate(onsets) if o is None]
    hallu = [paths[i] for i, o in enumerate(onsets) if o is not None]
    h_on = [o for o in onsets if o is not None]
    rows = sweep(clean, hallu, h_on, grid)
    res = {}
    for g in targets:
        op = at_arl0(rows, g)
        res[str(g)] = ({"edd": op["delay_edd"], "delay": op["delay"],
                        "recall": op["recall"], "arl0": op["arl0"]} if op else None)
    line = "  ".join(
        f"ARL0={g}: EDD={res[str(g)]['edd']:.1f}/rec={res[str(g)]['recall']:.2f}"
        for g in targets if res[str(g)])
    print(f"  {name:16s} {line}", flush=True)
    return res


def max_channel_paths(p1, p2):
    return [np.maximum(a, b) for a, b in zip(p1, p2)]


def main():
    fwd = json.load(open(PROBS))["ForwardGRU"]
    fwd_labs = [np.asarray(l, float) for l in fwd["labs"]]
    ref, _, _ = cusum_reference_value(fwd["probs"], fwd["labs"])

    data = load_and_enrich_all()
    te = assemble_features(data["te_base"], data["te_nli"], data["te_lm"], True, True)
    te_labs = [np.asarray(l, float) for l in data["te_labs"]]
    tr = assemble_features(data["tr_base"], data["tr_nli"], data["tr_lm"], True, True)
    Xtr = np.vstack([np.asarray(f) for f in tr]).astype(np.float64)
    ytr = np.concatenate([np.asarray(l) for l in data["tr_labs"]])

    # alignment check: same docs in same order
    assert len(te) == len(fwd_labs), f"{len(te)} feature docs vs {len(fwd_labs)} prob docs"
    mism = sum(len(te_labs[i]) != len(fwd_labs[i]) for i in range(len(te)))
    assert mism == 0, f"{mism} docs with length mismatch -- not aligned"
    print(f"Aligned {len(te)} test docs.", flush=True)

    clean, hallu = Xtr[ytr < 0.5], Xtr[ytr > 0.5]
    mu0, mu1 = clean.mean(0), hallu.mean(0)
    S0, S1 = shrink_cov(clean), shrink_cov(hallu)
    S0inv, S1inv = np.linalg.inv(S0), np.linalg.inv(S1)
    _, ld0 = np.linalg.slogdet(S0)
    _, ld1 = np.linalg.slogdet(S1)

    def gauss_inc(X):
        Z0, Z1 = X - mu0, X - mu1
        q0 = np.einsum("ij,jk,ik->i", Z0, S0inv, Z0)
        q1 = np.einsum("ij,jk,ik->i", Z1, S1inv, Z1)
        return 0.5 * (ld0 - ld1 + q0 - q1)

    def logit_inc(probs):
        p = np.clip(np.asarray(probs, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)) - ref

    onsets = [first_onset(l) for l in te_labs]
    logit_incs = [logit_inc(p) for p in fwd["probs"]]
    gauss_incs = [gauss_inc(np.asarray(X, float)) for X in te]

    # standardize each increment to unit clean-stream std (so the sum is balanced)
    def clean_std(incs):
        v = np.concatenate([incs[i] for i, o in enumerate(onsets) if o is None])
        return v.std() + 1e-8
    sL, sG = clean_std(logit_incs), clean_std(gauss_incs)
    logit_z = [a / sL for a in logit_incs]
    gauss_z = [a / sG for a in gauss_incs]

    p_logit = [cusum(a) for a in logit_z]
    p_gauss = [cusum(a) for a in gauss_z]
    p_sum = [cusum(a + b) for a, b in zip(logit_z, gauss_z)]

    print("\nDetectors (censored EDD / recall) at matched ARL0:")
    out = {}
    out["logit"] = evaluate("logit-CUSUM", p_logit, onsets, np.linspace(0, 300, 2001))
    out["gauss"] = evaluate("cov-Gaussian", p_gauss, onsets, np.linspace(0, 300, 2001))
    out["sum"] = evaluate("sum (L+G)", p_sum, onsets, np.linspace(0, 300, 2001))
    out["max"] = evaluate("max-channel", max_channel_paths(p_logit, p_gauss), onsets,
                          np.linspace(0, 300, 2001))
    print("\nReading: a combined rule that keeps recall ~0.43 AND delay ~11 would be a "
          "clean deployable gain -- high recall from covariance, speed from the scalar.")
    json.dump(out, open("combined_detector.json", "w"), indent=2)
    print("Saved -> combined_detector.json")


if __name__ == "__main__":
    main()
