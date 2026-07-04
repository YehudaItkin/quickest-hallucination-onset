#!/usr/bin/env python3
"""Adversarial ablation of the covariance-Gaussian detector before it goes in the paper.

The headline claim: a full-covariance Gaussian LLR CUSUM beats the learned scalar on
recall-honest censored EDD (recall 0.44 vs 0.24 at ARL0=100). Before believing it, kill
the obvious confounds:

  raw         : the detector as run -- inc = 0.5(logdet0 - logdet1 + q0 - q1), NO centering.
                The constant 0.5(logdet0 - logdet1) is a per-token DRIFT. If recall comes
                from that drift (a 'fire later' position effect) rather than the content,
                the result is spurious.
  centered    : subtract the train clean-token mean of the increment -> removes any constant
                drift. If recall survives, the signal is in the per-token content.
  midpoint    : subtract k = (clean_mean + hallu_mean)/2, the textbook CUSUM reference.
  quad-only   : the COVARIANCE term alone, 0.5 x'(S0^{-1} - S1^{-1}) x, centered. Is the
                advantage actually the covariance reshaping (the claim)?
  linear-only : the MEAN term alone, (S1^{-1}mu1 - S0^{-1}mu0)'x, centered. Or is it the mean?
  const-drift : a pure constant increment (= the logdet drift), centered-then-NOT, as a
                position-only null. Should detect nothing if centering is honest.

Also re-runs the scalar logit-CUSUM on the SAME threshold grid for a fair operating point.
Pure numpy; run in ~/hallucination_exp.
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
GRID = np.linspace(0, 400, 2001)


def shrink_cov(X):
    S = np.cov(X, rowvar=False)
    return (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S)) + 1e-6 * np.eye(S.shape[0])


def cusum(inc):
    S, p = 0.0, np.empty(len(inc))
    for t in range(len(inc)):
        S = max(0.0, S + inc[t]); p[t] = S
    return p


def op_at(incs, onsets, k):
    paths = [cusum(a - k) for a in incs]
    clean = [paths[i] for i, o in enumerate(onsets) if o is None]
    hallu = [paths[i] for i, o in enumerate(onsets) if o is not None]
    h_on = [o for o in onsets if o is not None]
    return at_arl0(sweep(clean, hallu, h_on, GRID), ARL0)


def main():
    fwd = json.load(open(PROBS))["ForwardGRU"]
    ref, _, _ = cusum_reference_value(fwd["probs"], fwd["labs"])
    data = load_and_enrich_all()
    tr = [np.asarray(f, float) for f in
          assemble_features(data["tr_base"], data["tr_nli"], data["tr_lm"], True, True)]
    te = [np.asarray(f, float) for f in
          assemble_features(data["te_base"], data["te_nli"], data["te_lm"], True, True)]
    Xtr = np.vstack(tr); ytr = np.concatenate([np.asarray(l) for l in data["tr_labs"]])
    te_labs = [np.asarray(l, float) for l in data["te_labs"]]
    onsets = [first_onset(l) for l in te_labs]

    c, h = Xtr[ytr < 0.5], Xtr[ytr > 0.5]
    mu0, mu1 = c.mean(0), h.mean(0)
    S0, S1 = shrink_cov(c), shrink_cov(h)
    S0i, S1i = np.linalg.inv(S0), np.linalg.inv(S1)
    _, ld0 = np.linalg.slogdet(S0); _, ld1 = np.linalg.slogdet(S1)
    Q = S0i - S1i                      # quadratic (covariance) coefficient
    b = S1i @ mu1 - S0i @ mu0          # linear (mean) coefficient

    def full(X):
        Z0, Z1 = X - mu0, X - mu1
        q0 = np.einsum("ij,jk,ik->i", Z0, S0i, Z0)
        q1 = np.einsum("ij,jk,ik->i", Z1, S1i, Z1)
        return 0.5 * (ld0 - ld1 + q0 - q1)

    def quad(X):
        return 0.5 * np.einsum("ij,jk,ik->i", X, Q, X)

    def linear(X):
        return X @ b

    def const(X):
        return np.full(len(X), 0.5 * (ld0 - ld1))

    variants = {"full": full, "quad-only": quad, "linear-only": linear, "const-drift": const}

    # train-estimated clean and hallu means of each variant (for centering)
    def train_means(fn):
        vals_c = np.concatenate([fn(d)[l < 0.5] for d, l in zip(tr, data["tr_labs"])
                                 if (np.asarray(l) < 0.5).any()])
        vals_h = np.concatenate([fn(d)[np.asarray(l) > 0.5] for d, l in zip(tr, data["tr_labs"])
                                 if (np.asarray(l) > 0.5).any()])
        return float(vals_c.mean()), float(vals_h.mean())

    print(f"Adversarial ablation, ARL0={ARL0} (censored EDD / recall):")
    out = {}
    for vname, fn in variants.items():
        te_inc = [fn(X) for X in te]
        mc, mh = train_means(fn)
        kmid = 0.5 * (mc + mh)
        # raw (no centering) only meaningful for 'full' and 'const'
        rows = {}
        for cname, k in [("centered", mc), ("midpoint", kmid)]:
            op = op_at(te_inc, onsets, k)
            rows[cname] = op
            tag = f"EDD={op['delay_edd']:.1f}/r={op['recall']:.2f}" if op else "n/a"
            print(f"  {vname:12s} [{cname:8s} k={k:7.2f}]: {tag}", flush=True)
        if vname in ("full", "const-drift"):
            op = op_at(te_inc, onsets, 0.0)   # NO centering -> raw drift
            rows["raw_k0"] = op
            tag = f"EDD={op['delay_edd']:.1f}/r={op['recall']:.2f}" if op else "n/a"
            print(f"  {vname:12s} [raw k=0.00 ]: {tag}  <-- includes the constant drift", flush=True)
        out[vname] = {k: (v if v else None) for k, v in rows.items()}

    # scalar logit baseline on the SAME grid + onset-position sanity
    lg = [cusum_path(p, ref) for p in fwd["probs"]]
    clean = [lg[i] for i, o in enumerate(onsets) if o is None]
    hallu = [lg[i] for i, o in enumerate(onsets) if o is not None]
    h_on = [o for o in onsets if o is not None]
    op = at_arl0(sweep(clean, hallu, h_on, GRID), ARL0)
    print(f"\n  logit-CUSUM (same grid): EDD={op['delay_edd']:.1f}/r={op['recall']:.2f}")
    doclen = np.mean([len(l) for l, o in zip(te_labs, onsets) if o is not None])
    onpos = np.mean([o for o in onsets if o is not None])
    print(f"  sanity: mean hallu-doc length {doclen:.0f}, mean onset position {onpos:.0f} "
          f"(if onsets are late, a drift detector cheats)")
    out["logit"] = op
    out["doc_stats"] = {"mean_len": float(doclen), "mean_onset": float(onpos)}

    print("\nVERDICT: if 'full [centered]' keeps recall ~0.44 and 'quad-only [centered]' "
          "carries most of it, the covariance signal is real. If recall lives only in "
          "'full [raw k=0]' / 'const-drift', it is a position/drift artifact -> DO NOT publish.")
    json.dump(out, open("covariance_ablate.json", "w"), indent=2)
    print("Saved -> covariance_ablate.json")


if __name__ == "__main__":
    main()
