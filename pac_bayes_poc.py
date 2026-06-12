#!/usr/bin/env python3
"""Sample-complexity view of why a temporal model generalizes: the tractable core
of a PAC-Bayes argument, computed from the label sequence alone.

A per-token i.i.d. learner treats the N tokens as N independent samples. They are
not: the label chain is strongly persistent (q = P(1->1) = 0.907), so it mixes
slowly and the effective number of independent samples is far below N. We quantify
this and the irreducible cost an i.i.d. model pays for ignoring the dependence.

What is NOT here: a full PAC-Bayes generalization bound for the recurrent model.
That needs the weight posterior / training process and is left as future work; this
script establishes the data-side quantities such a bound would rest on.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROBS = HERE / "directional_probs_seed42.json"
OUT = HERE / "pac_bayes_poc.json"


def hb(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def transition_matrix(seqs):
    c = np.zeros((2, 2))
    for s in seqs:
        for i in range(1, len(s)):
            c[int(s[i - 1]), int(s[i])] += 1
    rs = c.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return c / rs


def label_autocorr_time(seqs, max_lag=40):
    """Integrated autocorrelation time of the binary label sequence."""
    seqs = [s for s in seqs if len(s) > max_lag + 1]
    allv = np.concatenate(seqs)
    mu, var = allv.mean(), allv.var()
    rhos = []
    for k in range(1, max_lag + 1):
        num, n = 0.0, 0
        for s in seqs:
            num += float(np.sum((s[:-k] - mu) * (s[k:] - mu)))
            n += len(s) - k
        rho = (num / n) / var if n > 0 and var > 0 else 0.0
        if rho <= 0:
            break
        rhos.append(rho)
    return 1.0 + 2.0 * sum(rhos), rhos


def main():
    d = json.load(open(PROBS))
    seqs = [np.asarray(l) for l in d["ForwardGRU"]["labs"]]
    N = int(sum(len(s) for s in seqs))

    T = transition_matrix(seqs)
    p, q = float(T[0, 1]), float(T[1, 1])
    # stationary distribution
    pi1 = p / (p + (1 - q))
    pi0 = 1 - pi1
    H_marg = hb(pi1)
    # entropy rate of the Markov chain
    h_rate = pi0 * hb(p) + pi1 * hb(q)
    redundancy = 1 - h_rate / H_marg
    # second eigenvalue and mixing time
    lam2 = q - p
    mix = 1.0 / (1.0 - lam2)

    print(f"Label chain: p=P(0->1)={p:.4f}, q=P(1->1)={q:.4f}, persistence q/p={q/p:.0f}")
    print(f"Stationary pi1={pi1:.4f} (= base rate)\n")

    print("1. Redundancy an i.i.d. model ignores:")
    print(f"   marginal entropy H(Y)   = {H_marg:.4f} bits/token")
    print(f"   entropy rate h          = {h_rate:.4f} bits/token")
    print(f"   temporal redundancy     = 1 - h/H(Y) = {redundancy:.1%}")
    print(f"   -> a Bernoulli model is 'surprised' at {H_marg:.3f} bits/token when the")
    print(f"      process actually produces {h_rate:.3f}; it never models the dependence.\n")

    print("2. Effective sample size (slow mixing):")
    print(f"   second eigenvalue lambda2 = q-p = {lam2:.4f}")
    print(f"   mixing time ~ 1/(1-lambda2) = {mix:.1f} tokens")
    tau_Y, _ = label_autocorr_time(seqs)
    n_spans = sum(int(((s[1:] > 0.5) & (s[:-1] < 0.5)).sum() + (s[0] > 0.5)) for s in seqs)
    print(f"   integrated autocorr time tau_Y = {tau_Y:.1f}")
    print(f"   N tokens = {N:,};  N_eff ~ N/tau_Y = {N/tau_Y:,.0f}")
    print(f"   independent 'events' (spans) = {n_spans:,}")
    print(f"   -> the i.i.d. learner overcounts its samples by ~{tau_Y:.0f}x.\n")

    print("3. Sample-complexity separation (informal):")
    print("   The temporal class needs 2 parameters (p, q) to capture the dynamics")
    print("   and attains entropy rate h; the i.i.d. (Bernoulli) class has 1 parameter")
    print("   but an irreducible excess surprise of H(Y) - h = "
          f"{H_marg - h_rate:.3f} bits/token, regardless of sample size. Capacity")
    print("   cannot fix a model class that omits the dependence; the temporal model")
    print("   wins on approximation error, not just estimation error.")
    print("\nNOTE: a full PAC-Bayes generalization bound for the recurrent model needs")
    print("its weight posterior and the training process (a server experiment). This")
    print("establishes only the data-side quantities (N_eff, redundancy, separation).")

    json.dump({
        "p": p, "q": q, "pi1": pi1, "H_marginal": H_marg, "entropy_rate": h_rate,
        "redundancy": redundancy, "lambda2": lam2, "mixing_time": mix,
        "tau_Y": tau_Y, "N_tokens": N, "N_eff": N / tau_Y, "n_spans": n_spans,
        "excess_surprise_iid": H_marg - h_rate,
    }, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
