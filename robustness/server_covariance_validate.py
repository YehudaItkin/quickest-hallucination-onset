#!/usr/bin/env python3
"""Validate the covariance-Gaussian onset detector before claiming it.

Direction-1 found a full-covariance Gaussian LLR detector dominates the learned scalar
on the recall-honest metric (censored EDD 45 vs 62, recall 0.44 vs 0.24 at ARL0=100).
Two checks before it goes in the paper:

  V1 stability  : the mean/covariance are estimated on train. Bootstrap that estimation
                  (resample train documents) and re-evaluate on the fixed test set. If
                  recall/EDD are stable, the advantage is not an estimation fluke.
  V2 cross-model: RAGTruth mixes six generators. Leave one generator OUT of the train
                  estimation and test on that generator's documents -- the covariance
                  detector is then out-of-distribution, while the saved ForwardGRU
                  (trained on all generators) is in-distribution. If the covariance
                  detector still beats the scalar per held-out generator, the signal is
                  generator-general, not a per-model covariance artifact.

Pure numpy on the 33-d features + the aligned ForwardGRU posteriors. Run in
~/hallucination_exp.
"""
import json
from pathlib import Path

import numpy as np

from run_extended import load_and_enrich_all, assemble_features
from run_learned_cusum import cusum_reference_value, cusum_path, first_onset, sweep, at_arl0

HERE = Path(__file__).resolve().parent
PROBS = HERE / "directional_probs_seed42.json"
ARL0 = 100
SHRINK = 0.1
GRID = np.linspace(0, 400, 1601)
LOGIT_GRID = np.linspace(0, 120, 801)


def shrink_cov(X):
    S = np.cov(X, rowvar=False)
    return (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S)) + 1e-6 * np.eye(S.shape[0])


def fit(Xtr, ytr):
    c, h = Xtr[ytr < 0.5], Xtr[ytr > 0.5]
    mu0, mu1 = c.mean(0), h.mean(0)
    S0, S1 = shrink_cov(c), shrink_cov(h)
    _, ld0 = np.linalg.slogdet(S0)
    _, ld1 = np.linalg.slogdet(S1)
    return mu0, mu1, np.linalg.inv(S0), np.linalg.inv(S1), ld0, ld1


def gauss_paths(docs, params):
    mu0, mu1, S0i, S1i, ld0, ld1 = params
    out = []
    for X in docs:
        Z0, Z1 = X - mu0, X - mu1
        q0 = np.einsum("ij,jk,ik->i", Z0, S0i, Z0)
        q1 = np.einsum("ij,jk,ik->i", Z1, S1i, Z1)
        inc = 0.5 * (ld0 - ld1 + q0 - q1)
        S, p = 0.0, np.empty(len(X))
        for t in range(len(X)):
            S = max(0.0, S + inc[t]); p[t] = S
        out.append(p)
    return out


def op_at(paths, onsets, grid, arl0=ARL0):
    clean = [paths[i] for i, o in enumerate(onsets) if o is None]
    hallu = [paths[i] for i, o in enumerate(onsets) if o is not None]
    h_on = [o for o in onsets if o is not None]
    if not hallu or not clean:
        return None
    return at_arl0(sweep(clean, hallu, h_on, grid), arl0)


def main():
    fwd = json.load(open(PROBS))["ForwardGRU"]
    ref, _, _ = cusum_reference_value(fwd["probs"], fwd["labs"])
    data = load_and_enrich_all()
    tr = [np.asarray(f, float) for f in
          assemble_features(data["tr_base"], data["tr_nli"], data["tr_lm"], True, True)]
    te = [np.asarray(f, float) for f in
          assemble_features(data["te_base"], data["te_nli"], data["te_lm"], True, True)]
    tr_labs = [np.asarray(l, float) for l in data["tr_labs"]]
    te_labs = [np.asarray(l, float) for l in data["te_labs"]]
    tr_model = [e["model"] for e in data["tr_ex"]]
    te_model = [e["model"] for e in data["te_ex"]]
    onsets = [first_onset(l) for l in te_labs]
    logit_paths = [cusum_path(p, ref) for p in fwd["probs"]]
    out = {}

    # ---- V1: bootstrap the train estimation, evaluate on fixed test ----
    print("V1 stability (bootstrap train docs, fixed test, ARL0=100):", flush=True)
    rng = np.random.RandomState(0)
    rec, edd = [], []
    nb = 5
    for b in range(nb):
        idx = rng.randint(0, len(tr), len(tr))
        Xb = np.vstack([tr[i] for i in idx]); yb = np.concatenate([tr_labs[i] for i in idx])
        op = op_at(gauss_paths(te, fit(Xb, yb)), onsets, GRID)
        if op:
            rec.append(op["recall"]); edd.append(op["delay_edd"])
            print(f"  boot {b}: recall={op['recall']:.2f}  censored EDD={op['delay_edd']:.1f}", flush=True)
    out["v1"] = {"recall_mean": float(np.mean(rec)), "recall_std": float(np.std(rec)),
                 "edd_mean": float(np.mean(edd)), "edd_std": float(np.std(edd))}
    print(f"  => recall {np.mean(rec):.2f}±{np.std(rec):.2f}, "
          f"EDD {np.mean(edd):.1f}±{np.std(edd):.1f}\n", flush=True)

    # ---- V2: leave-one-generator-out ----
    gens = sorted(set(te_model))
    print(f"V2 cross-model leave-one-generator-out (ARL0=100), {len(gens)} generators:")
    print(f"  {'generator':22s} {'cov-Gauss(LOO)':>16s} {'ForwardGRU(in-dist)':>22s}")
    out["v2"] = {}
    for g in gens:
        tr_keep = [i for i, m in enumerate(tr_model) if m != g]
        te_g = [i for i, m in enumerate(te_model) if m == g]
        Xk = np.vstack([tr[i] for i in tr_keep]); yk = np.concatenate([tr_labs[i] for i in tr_keep])
        params = fit(Xk, yk)
        og = [onsets[i] for i in te_g]
        cov_op = op_at(gauss_paths([te[i] for i in te_g], params), og, GRID)
        lg_op = op_at([logit_paths[i] for i in te_g], og, LOGIT_GRID)
        cov_s = f"EDD={cov_op['delay_edd']:.1f}/r={cov_op['recall']:.2f}" if cov_op else "n/a"
        lg_s = f"EDD={lg_op['delay_edd']:.1f}/r={lg_op['recall']:.2f}" if lg_op else "n/a"
        n_hallu = sum(o is not None for o in og)
        print(f"  {g:22s} {cov_s:>16s} {lg_s:>22s}   (hallu docs {n_hallu})", flush=True)
        out["v2"][g] = {"cov": cov_op, "logit": lg_op, "n_hallu": n_hallu}

    print("\nReading: V1 stable (small std) => not an estimation fluke. V2 cov-Gauss(LOO) "
          "below ForwardGRU(in-dist) per generator => the covariance onset signal is "
          "generator-general, even out-of-distribution.")
    json.dump(out, open("covariance_validate.json", "w"), indent=2)
    print("Saved -> covariance_validate.json")


if __name__ == "__main__":
    main()
