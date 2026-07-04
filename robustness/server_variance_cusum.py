#!/usr/bin/env python3
"""Direction 1: detect the change in DISPERSION, not the mean.

The multivariate result (Appendix F) found the hallucination change is barely a mean
shift: the raw-feature Mahalanobis separation is 0.40 nats against the 3.5 the
diagonal-Gaussian divergence assigns -- the rest is variance and shape. A drift CUSUM
exploits only the mean and throws that away. Here we target it directly.

  energy-CUSUM   : score each token by its Mahalanobis energy under the CLEAN model,
                   e_t = (x_t-mu0)' Sigma0^{-1} (x_t-mu0). Under faithful tokens
                   e_t ~ chi^2_d (mean d); a dispersion increase shifts its mean up, so
                   a CUSUM on (e_t - d) detects the variance change.
  full-Gaussian  : the complete Gaussian log-likelihood-ratio increment for
                   N(mu0,Sigma0) -> N(mu1,Sigma1) with FULL covariances (mean AND
                   covariance change), accumulated as a CUSUM. This is the naive
                   Gaussian CUSUM done right -- full covariance instead of diagonal.

Compared at matched ARL0 to the scalar ForwardGRU-CUSUM (11.5) and the floor. If a
variance-aware detector beats the drift CUSUM, hallucination onset is better seen as a
dispersion change than a mean shift -- a reframing of the detection target. Pure numpy
on the 33-d features; run in ~/hallucination_exp.
"""
import json

import numpy as np

from run_extended import load_and_enrich_all, assemble_features
from run_learned_cusum import first_onset, sweep, at_arl0

GAMMA_TARGETS = [50, 100, 200]
ARL0_MAIN = 100
SHRINK = 0.1


def shrink_cov(X):
    S = np.cov(X, rowvar=False)
    return (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S)) + 1e-6 * np.eye(S.shape[0])


def cusum_from_increment(docs, inc_fn):
    out = []
    for X in docs:
        inc = inc_fn(X)
        S, path = 0.0, np.empty(len(X))
        for t in range(len(X)):
            S = max(0.0, S + inc[t])
            path[t] = S
        out.append(path)
    return out


def evaluate(name, paths, onsets, grid):
    clean = [paths[i] for i, o in enumerate(onsets) if o is None]
    hallu = [paths[i] for i, o in enumerate(onsets) if o is not None]
    h_on = [o for o in onsets if o is not None]
    rows = sweep(clean, hallu, h_on, grid)
    res = {}
    for g in GAMMA_TARGETS:
        op = at_arl0(rows, g)
        res[str(g)] = ({"edd": op["delay_edd"], "delay": op["delay"],
                        "recall": op["recall"], "arl0": op["arl0"]} if op else None)
    op = res[str(ARL0_MAIN)]
    if op:
        print(f"  {name:16s} ARL0=100: censored EDD={op['edd']:.1f}  delay={op['delay']:.1f}  "
              f"recall={op['recall']:.2f}  @ARL0={op['arl0']:.0f}", flush=True)
    return res


def main():
    data = load_and_enrich_all()
    assert data["has_all"]
    tr = assemble_features(data["tr_base"], data["tr_nli"], data["tr_lm"], True, True)
    te = assemble_features(data["te_base"], data["te_nli"], data["te_lm"], True, True)
    Xtr = np.vstack([np.asarray(f) for f in tr]).astype(np.float64)
    ytr = np.concatenate([np.asarray(l) for l in data["tr_labs"]])
    te_docs = [np.asarray(f, dtype=np.float64) for f in te]
    onsets = [first_onset(np.asarray(l)) for l in data["te_labs"]]
    d = Xtr.shape[1]

    clean, hallu = Xtr[ytr < 0.5], Xtr[ytr > 0.5]
    mu0, mu1 = clean.mean(0), hallu.mean(0)
    S0, S1 = shrink_cov(clean), shrink_cov(hallu)
    S0inv, S1inv = np.linalg.inv(S0), np.linalg.inv(S1)
    sign0, logdet0 = np.linalg.slogdet(S0)
    sign1, logdet1 = np.linalg.slogdet(S1)
    var_ratio = float(np.trace(S1 @ S0inv) / d)
    print(f"dim={d}; clean var vs hallu: tr(S1 S0^-1)/d = {var_ratio:.2f} "
          f"(>1 => hallucinated tokens are more dispersed)", flush=True)

    def energy_inc(X):
        Z = X - mu0
        e = np.einsum("ij,jk,ik->i", Z, S0inv, Z)
        return e - d                                  # centered at the clean chi^2 mean

    def gauss_llr_inc(X):
        Z0, Z1 = X - mu0, X - mu1
        q0 = np.einsum("ij,jk,ik->i", Z0, S0inv, Z0)
        q1 = np.einsum("ij,jk,ik->i", Z1, S1inv, Z1)
        return 0.5 * (logdet0 - logdet1 + q0 - q1)    # log N(x;1)/N(x;0)

    print("\nVariance-aware detectors at matched ARL0:")
    out = {"var_ratio": var_ratio}
    out["energy-CUSUM"] = evaluate("energy-CUSUM",
                                   cusum_from_increment(te_docs, energy_inc), onsets,
                                   np.linspace(0, 400, 901))
    out["full-Gaussian"] = evaluate("full-Gaussian",
                                    cusum_from_increment(te_docs, gauss_llr_inc), onsets,
                                    np.linspace(0, 400, 901))
    print("\nReference: ForwardGRU scalar-CUSUM 11.5 tok, naive DIAGONAL-Gaussian ~41, floor 1.3.")
    print("Reading: a variance-aware detector below 41 (and near/under 11.5) => onset is "
          "better seen as a dispersion change than a mean shift.")
    json.dump(out, open("variance_cusum.json", "w"), indent=2)
    print("Saved -> variance_cusum.json")


if __name__ == "__main__":
    main()
