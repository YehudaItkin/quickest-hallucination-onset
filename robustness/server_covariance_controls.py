#!/usr/bin/env python3
"""The two controls the critics demanded before the covariance edge can be claimed.

C1 symmetric cross-model: leave one generator OUT of BOTH detectors (the earlier V2
   was unfair -- HistGBM/ForwardGRU had seen the held-out generator). Refit cov-Gauss
   AND HistGBM without generator g, test on g. Report recall + delay-among-detected.
C2 paired bootstrap: is the +0.05 recall edge over HistGBM at ARL0=100 significant?
   Fix each detector's threshold at the full-data ARL0=100 operating point, resample
   the hallucination test documents (1000x), and report the 95% CI of the recall
   DIFFERENCE cov(quad) - HistGBM-threshold.

Pure numpy + sklearn HistGBM. Run in ~/hallucination_exp.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from run_extended import load_and_enrich_all, assemble_features
from run_learned_cusum import first_onset, first_crossing, sweep, at_arl0, arl0_on_clean_stream

HERE = Path(__file__).resolve().parent
SHRINK = 0.1
GRID = np.linspace(0, 400, 2001)
PROB_GRID = np.unique(np.concatenate([np.linspace(0, 0.999, 600), 1 - np.logspace(-6, -1, 300)]))
ARL0 = 100
HGB = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, random_state=0)


def shrink_cov(X):
    S = np.cov(X, rowvar=False)
    return (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S)) + 1e-6 * np.eye(S.shape[0])


def cusum(inc):
    S, p = 0.0, np.empty(len(inc))
    for t in range(len(inc)):
        S = max(0.0, S + inc[t]); p[t] = S
    return p


def fit_quad(Xk, yk, tr_docs, tr_labs, keep):
    c, h = Xk[yk < 0.5], Xk[yk > 0.5]
    Q = np.linalg.inv(shrink_cov(c)) - np.linalg.inv(shrink_cov(h))
    quad = lambda X: 0.5 * np.einsum("ij,jk,ik->i", X, Q, X)
    qc = np.concatenate([quad(tr_docs[i])[tr_labs[i] < 0.5] for i in keep
                         if (tr_labs[i] < 0.5).any()])
    qh = np.concatenate([quad(tr_docs[i])[tr_labs[i] > 0.5] for i in keep
                         if (tr_labs[i] > 0.5).any()])
    k = 0.5 * (qc.mean() + qh.mean())
    return lambda docs: [cusum(quad(X) - k) for X in docs]


def op(paths, onsets, grid, arl0=ARL0):
    clean = [paths[i] for i, o in enumerate(onsets) if o is None]
    hallu = [paths[i] for i, o in enumerate(onsets) if o is not None]
    h_on = [o for o in onsets if o is not None]
    if not hallu or not clean:
        return None
    return at_arl0(sweep(clean, hallu, h_on, grid), arl0)


def main():
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
    out = {}

    # ---------------- C1: symmetric leave-one-generator-out ----------------
    print("C1 symmetric LOO (both refit without the generator), ARL0=100:")
    print(f"  {'generator':22s} {'cov-Gauss(quad)':>20s} {'HistGBM-thresh':>20s}")
    out["c1"] = {}
    for g in sorted(set(te_model)):
        keep = [i for i, m in enumerate(tr_model) if m != g]
        te_g = [i for i, m in enumerate(te_model) if m == g]
        og = [onsets[i] for i in te_g]
        Xk = np.vstack([tr[i] for i in keep]); yk = np.concatenate([tr_labs[i] for i in keep])
        # cov-Gauss quad
        det = fit_quad(Xk, yk, tr, tr_labs, keep)
        cov_op = op(det([te[i] for i in te_g]), og, GRID)
        # HistGBM threshold (refit without g)
        clf = HistGradientBoostingClassifier(**HGB).fit(Xk, (yk > 0.5).astype(int))
        hp = [clf.predict_proba(te[i])[:, 1] for i in te_g]
        hist_op = op([np.asarray(p) for p in hp], og, PROB_GRID)
        cs = f"r={cov_op['recall']:.2f}/d={cov_op['delay']:.1f}" if cov_op else "n/a"
        hs = f"r={hist_op['recall']:.2f}/d={hist_op['delay']:.1f}" if hist_op else "n/a"
        n = sum(o is not None for o in og)
        print(f"  {g:22s} {cs:>20s} {hs:>20s}   (hallu {n})", flush=True)
        out["c1"][g] = {"cov": cov_op, "hist": hist_op, "n_hallu": n}

    # ---------------- C2: paired bootstrap of the recall difference ----------------
    print("\nC2 paired bootstrap of recall difference (cov - HistGBM) at ARL0=100:")
    Xtr = np.vstack(tr); ytr = np.concatenate(tr_labs)
    keep_all = list(range(len(tr)))
    det = fit_quad(Xtr, ytr, tr, tr_labs, keep_all)
    cov_paths = det(te)
    clf = HistGradientBoostingClassifier(**HGB).fit(Xtr, (ytr > 0.5).astype(int))
    hist_paths = [np.asarray(clf.predict_proba(X)[:, 1]) for X in te]

    # fix each detector's threshold at full-data ARL0=100
    def threshold_at(paths, grid):
        clean = [paths[i] for i, o in enumerate(onsets) if o is None]
        rows = [(t, arl0_on_clean_stream(clean, t)) for t in grid]
        above = [(t, a) for t, a in rows if a >= ARL0]
        return min(above, key=lambda r: r[1])[0] if above else grid[-1]
    h_cov = threshold_at(cov_paths, GRID)
    h_hist = threshold_at(hist_paths, PROB_GRID)

    hallu_idx = [i for i, o in enumerate(onsets) if o is not None]
    def detected(paths, h):
        return np.array([1 if (first_crossing(paths[i], h) is not None
                               and first_crossing(paths[i], h) >= onsets[i]) else 0
                         for i in hallu_idx])
    d_cov, d_hist = detected(cov_paths, h_cov), detected(hist_paths, h_hist)
    print(f"  full-data recall: cov={d_cov.mean():.3f}  HistGBM={d_hist.mean():.3f}  "
          f"diff={d_cov.mean()-d_hist.mean():+.3f}")
    rng = np.random.RandomState(0)
    diffs = []
    n = len(hallu_idx)
    for _ in range(1000):
        bi = rng.randint(0, n, n)
        diffs.append(d_cov[bi].mean() - d_hist[bi].mean())
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  bootstrap recall diff: {diffs.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"(significant if CI excludes 0)")
    out["c2"] = {"recall_cov": float(d_cov.mean()), "recall_hist": float(d_hist.mean()),
                 "diff_mean": float(diffs.mean()), "ci": [float(lo), float(hi)]}

    json.dump(out, open("covariance_controls.json", "w"), indent=2)
    print("\nSaved -> covariance_controls.json")


if __name__ == "__main__":
    main()
