#!/usr/bin/env python3
"""Seed-robustness of the speedup decomposition.

Reviewer concern: the headline decomposition (nonlinear score / accumulation /
context) and its significance come from ONE trained model (seed 42); the bootstrap
CIs capture document variance but NOT training variance. This script retrains the
detectors under several seeds, recomputes the three decomposition drops at a matched
ARL0=100, and reports mean +/- std across seeds, so the "significant"/"within noise"
calls can be checked against training variance.

Drops (delay among detected, tokens):
  nonlinearity = LogReg-threshold   -> HistGBM-threshold
  accumulation = HistGBM-threshold  -> HistGBM-CUSUM
  context      = HistGBM-CUSUM      -> ForwardGRU-CUSUM

Run on the GPU server in ~/hallucination_exp.
"""
import json

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from run_extended import load_and_enrich_all, assemble_features, train_nn, set_seed
from run_directional_ablation import ForwardGRU
from nonlinear_baseline import delay_at_arl0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2, 7, 42]
TARGET = 100


def per_doc_proba(clf, te, transform=None):
    out = []
    for f in te:
        if len(f) == 0:
            out.append([])
            continue
        x = transform(f) if transform is not None else f
        out.append(clf.predict_proba(x)[:, 1].tolist())
    return out


def main():
    data = load_and_enrich_all()
    assert data["has_all"], "need all signals"
    tr = assemble_features(data["tr_base"], data["tr_nli"], data["tr_lm"], True, True)
    te = assemble_features(data["te_base"], data["te_nli"], data["te_lm"], True, True)
    dim = tr[0].shape[1]
    Xtr = np.vstack(tr)
    ytr = np.concatenate([np.asarray(l) for l in data["tr_labs"]])
    labs = data["te_labs"]
    hallu_flags = [int(ex["has_hallucination"]) for ex in data["tr_ex"]]

    drops = {"nonlinearity": [], "accumulation": [], "context": []}
    cells = {"LogReg-t": [], "HistGBM-t": [], "HistGBM-c": [], "ForwardGRU-c": []}

    for seed in SEEDS:
        set_seed(seed)
        tr_idx, val_idx = train_test_split(range(len(tr)), test_size=0.15,
                                           random_state=seed, stratify=hallu_flags)
        tr_f = [tr[i] for i in tr_idx]; tr_l = [data["tr_labs"][i] for i in tr_idx]
        val_f = [tr[i] for i in val_idx]; val_l = [data["tr_labs"][i] for i in val_idx]

        # ForwardGRU (recurrent, learned CUSUM)
        set_seed(seed)
        fg = ForwardGRU(dim=dim, h=64).to(DEVICE)
        fg_probs, _ = train_nn(fg, tr_f, tr_l, val_f, val_l, te, labs, DEVICE)

        # HistGBM (nonlinear per-token) on raw features
        hg = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                            max_leaf_nodes=63, validation_fraction=0.1,
                                            random_state=seed).fit(Xtr, ytr)
        hg_probs = per_doc_proba(hg, te)

        # LogReg (linear per-token) on standardized features
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        lr_probs = per_doc_proba(lr, te, transform=sc.transform)

        def d(probs, mode):
            op = delay_at_arl0(probs, labs, mode, TARGET)
            return op["delay"] if op else float("nan")

        lr_t = d(lr_probs, "threshold")
        hg_t = d(hg_probs, "threshold")
        hg_c = d(hg_probs, "cusum")
        fg_c = d(fg_probs, "cusum")
        cells["LogReg-t"].append(lr_t); cells["HistGBM-t"].append(hg_t)
        cells["HistGBM-c"].append(hg_c); cells["ForwardGRU-c"].append(fg_c)
        drops["nonlinearity"].append(lr_t - hg_t)
        drops["accumulation"].append(hg_t - hg_c)
        drops["context"].append(hg_c - fg_c)
        print(f"seed {seed}: LogReg-t {lr_t:.1f} HistGBM-t {hg_t:.1f} "
              f"HistGBM-c {hg_c:.1f} ForwardGRU-c {fg_c:.1f}", flush=True)

    print("\nSeed-averaged decomposition (delay among detected, ARL0=100):")
    out = {"seeds": SEEDS, "cells": {}, "drops": {}}
    for nm, vals in cells.items():
        v = np.array(vals, float)
        out["cells"][nm] = [float(v.mean()), float(v.std())]
        print(f"  {nm:14s}: {v.mean():5.1f} +/- {v.std():.1f}")
    print("\nDrops, mean +/- std (std now includes training-seed variance):")
    for nm, vals in drops.items():
        v = np.array(vals, float)
        sig = "robust (>1 sd from 0)" if abs(v.mean()) > v.std() else "fragile (within 1 sd of 0)"
        out["drops"][nm] = [float(v.mean()), float(v.std())]
        print(f"  {nm:14s}: {v.mean():+5.1f} +/- {v.std():.1f}  [{sig}]")

    json.dump(out, open("server_seed_decomposition.json", "w"), indent=2, default=float)
    print("\nSaved -> server_seed_decomposition.json")


if __name__ == "__main__":
    main()
