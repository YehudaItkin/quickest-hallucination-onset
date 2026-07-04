#!/usr/bin/env python3
"""Direction B: a MULTIVARIATE CUSUM to lift the scalar bottleneck.

The deficit decomposition (Appendix F / symmetry_poc) put the whole 4.5x loss in
D_feat/D_inc ~ 4: compressing the 33-d feature law into one scalar log-odds throws
away divergence. A scalar accumulator cannot recover it; a multivariate one might.

Two feature-space detectors, both compared at matched ARL0 against the scalar
baselines (LogReg-CUSUM, ForwardGRU-CUSUM) and the Lorden floor:

  LDA-CUSUM   -- the best FIXED linear projection. increment = w'(x) - k,
                 w = Sigma^{-1}(mu1-mu0), k = w'(mu0+mu1)/2. Full pre-change
                 covariance (shrinkage-regularized), unlike the diagonal naive
                 Gaussian CUSUM. With a KNOWN change direction this is the optimal
                 *linear* statistic -- still scalar, so it bounds what one projection
                 can do.
  GLR-CUSUM   -- window-limited generalized likelihood ratio for a change in mean
                 of UNKNOWN direction (Hotelling form). At each t it maximizes over
                 a putative change-point s in [t-w, t):
                     GLR_t = max_s  ||Z_t - Z_s||^2 / (2 (t-s)),
                 Z = cumsum of the whitened deviations Sigma^{-1/2}(x - mu0). This
                 is genuinely multivariate: it adapts the direction to the data, so
                 it beats the fixed LDA projection iff the hallucination shift is
                 heterogeneous in direction.

If GLR > LDA > scalar, multivariate accumulation lifts the D_feat/D_inc bottleneck.
If GLR ~ LDA, the change is effectively one-directional and the scalar bottleneck is
intrinsic. Pure numpy on the 33-d features; no GPU. Run in ~/hallucination_exp.
"""
import json

import numpy as np

from run_extended import load_and_enrich_all, assemble_features
from run_learned_cusum import first_onset, sweep, at_arl0

GAMMA_TARGETS = [50, 100, 200]
ARL0_MAIN = 100
SHRINK = 0.1          # covariance shrinkage toward diagonal (regularizes Sigma^{-1})
GLR_WINDOW = 50       # window-limited GLR look-back (tokens)
D_FEATURE = 3.5       # paper's diagonal-Gaussian feature divergence (nats)


def whiten_stats(Xtr, ytr):
    """Estimate mu0, mu1, and a shrinkage pre-change covariance on train tokens."""
    clean, hallu = Xtr[ytr < 0.5], Xtr[ytr > 0.5]
    mu0, mu1 = clean.mean(0), hallu.mean(0)
    S = np.cov(clean, rowvar=False)
    S = (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S)) + 1e-6 * np.eye(S.shape[0])
    return mu0, mu1, S


def lda_paths(docs, mu0, mu1, Sinv):
    """Per-document CUSUM path of the fixed LDA projection (scalar, optimal linear)."""
    w = Sinv @ (mu1 - mu0)
    k = float(w @ (mu0 + mu1) / 2.0)
    out = []
    for X in docs:
        inc = X @ w - k
        S, path = 0.0, np.empty(len(X))
        for t in range(len(X)):
            S = max(0.0, S + inc[t])
            path[t] = S
        out.append(path)
    return out


def glr_paths(docs, mu0, Whalf, window):
    """Per-document window-limited multivariate GLR statistic path (Hotelling form)."""
    out = []
    for X in docs:
        Z = np.vstack([np.zeros(X.shape[1]),
                       np.cumsum((X - mu0) @ Whalf.T, axis=0)])   # Z[0..T], Z[t]=sum_{1..t}
        T = len(X)
        path = np.empty(T)
        for t in range(1, T + 1):
            s0 = max(0, t - window)
            s = np.arange(s0, t)
            diff = Z[t] - Z[s0:t]                 # (t-s) x d
            num = np.einsum("ij,ij->i", diff, diff)
            stat = num / (2.0 * (t - s))
            path[t - 1] = stat.max()
        out.append(path)
    return out


def evaluate(name, paths, onsets, grids):
    clean_idx = [i for i, o in enumerate(onsets) if o is None]
    hallu_idx = [i for i, o in enumerate(onsets) if o is not None]
    rows = sweep([paths[i] for i in clean_idx], [paths[i] for i in hallu_idx],
                 [onsets[i] for i in hallu_idx], grids)
    res = {}
    for g in GAMMA_TARGETS:
        op = at_arl0(rows, g)
        res[str(g)] = ({"edd": op["delay_edd"], "delay": op["delay"],
                        "recall": op["recall"], "arl0": op["arl0"]} if op else None)
    op = res[str(ARL0_MAIN)]
    if op:
        print(f"  {name:14s} ARL0=100: censored EDD={op['edd']:.1f}  "
              f"delay={op['delay']:.1f}  recall={op['recall']:.2f}  @ARL0={op['arl0']:.0f}",
              flush=True)
    return res


def main():
    data = load_and_enrich_all()
    assert data["has_all"], "need Text+NLI+LM 33-d"
    tr = assemble_features(data["tr_base"], data["tr_nli"], data["tr_lm"], True, True)
    te = assemble_features(data["te_base"], data["te_nli"], data["te_lm"], True, True)
    Xtr = np.vstack([np.asarray(f) for f in tr]).astype(np.float64)
    ytr = np.concatenate([np.asarray(l) for l in data["tr_labs"]])
    te_docs = [np.asarray(f, dtype=np.float64) for f in te]
    onsets = [first_onset(np.asarray(l)) for l in data["te_labs"]]
    n_hallu = sum(o is not None for o in onsets)
    print(f"Train {len(ytr):,} tok ({int((ytr>0.5).sum()):,} hallu); "
          f"test {len(te_docs)} docs ({n_hallu} hallu). dim={Xtr.shape[1]}", flush=True)

    mu0, mu1, S = whiten_stats(Xtr, ytr)
    Sinv = np.linalg.inv(S)
    # symmetric inverse square root Sigma^{-1/2} for whitening
    evals, evecs = np.linalg.eigh(S)
    Whalf = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    maha = float((mu1 - mu0) @ Sinv @ (mu1 - mu0))
    print(f"Mahalanobis separation (mu1-mu0)'S^-1(mu1-mu0) = {maha:.3f} "
          f"(full-cov feature divergence ~ {maha/2:.2f} nats)", flush=True)

    lda_grid = np.linspace(0.0, 200.0, 801)
    glr_grid = np.linspace(0.0, 80.0, 801)

    print("\nDetectors at matched ARL0 (censored EDD / delay-among-detected / recall):")
    out = {"maha": maha, "d_feature": D_FEATURE}
    out["LDA-CUSUM"] = evaluate("LDA-CUSUM", lda_paths(te_docs, mu0, mu1, Sinv), onsets, lda_grid)
    out["GLR-CUSUM"] = evaluate(f"GLR-CUSUM(w{GLR_WINDOW})",
                                glr_paths(te_docs, mu0, Whalf, GLR_WINDOW), onsets, glr_grid)

    print(f"\nReference (paper, ARL0=100): ForwardGRU-CUSUM 11.5 tok, LogReg 30.8, "
          f"naive diagonal-Gaussian ~41, Lorden floor 1.3.")
    print("Reading: GLR > LDA > scalar => multivariate accumulation lifts the "
          "D_feat/D_inc scalar bottleneck; GLR ~ LDA => the change is one-directional.")
    json.dump(out, open("vector_cusum.json", "w"), indent=2)
    print("Saved -> vector_cusum.json")


if __name__ == "__main__":
    main()
