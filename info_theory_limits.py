#!/usr/bin/env python3
"""Information-theoretic view of the 0.845 ceiling.

How much of the per-token label information do the features carry, and does the
architecture-independent ceiling correspond to a saturated, feature-limited
mutual information?

From the saved posteriors we estimate the captured mutual information
  I(p_hat; Y) = H(Y) - H(Y | p_hat)
for three detectors that differ only in how much context they use:
  LogReg     -- per-token, no context
  ForwardGRU -- causal temporal context
  BiGRU      -- full (bidirectional) context

I(p_hat; Y) is a LOWER bound on the feature information I(X; Y) (data-processing:
Y -> X -> p_hat). So these numbers lower-bound what the 33-dim features contain,
and the increments quantify the value of temporal context. If the best
architecture (BiGRU) captures barely more than ForwardGRU and both sit far below
H(Y), the bottleneck is the features, not the model -- the information reading of
the architecture-independent ceiling. A definitive Bayes-error / Fano statement
needs the raw features and is left as a server experiment.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROBS = HERE / "directional_probs_seed42.json"
OUT = HERE / "info_theory_limits.json"


def hb(p):
    """Binary entropy in bits."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def captured_mi(p_hat, y, n_bins=40):
    """I(p_hat; Y) = H(Y) - H(Y | p_hat), equal-frequency binning of p_hat.

    H(Y|p_hat) uses the empirical P(Y=1 | bin), so it is calibration-free
    (it does not assume p_hat is calibrated).
    """
    pi = float(y.mean())
    H_Y = float(hb(pi))
    # equal-frequency bin edges
    qs = np.quantile(p_hat, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    idx = np.digitize(p_hat, qs[1:-1])
    H_cond = 0.0
    N = len(y)
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        p_y = float(y[m].mean())
        H_cond += (nb / N) * float(hb(p_y))
    I = H_Y - H_cond
    return {"H_Y": H_Y, "H_Y_given_phat": H_cond, "I": I,
            "efficiency": I / H_Y if H_Y > 0 else float("nan")}


def main():
    d = json.load(open(PROBS))
    y = np.concatenate([np.asarray(l) for l in d["ForwardGRU"]["labs"]])
    pi = float(y.mean())
    print(f"Per-token label: pi={pi:.4f}, H(Y)={float(hb(pi)):.4f} bits "
          f"(rare-event, low entropy)\n")

    models = ["LogReg", "ForwardGRU", "BiGRU"]
    res = {}
    print(f"{'detector':<12} | {'I(p_hat;Y) bits':>15} | {'eff = I/H(Y)':>12} | "
          f"{'H(Y|p_hat)':>10}")
    print("-" * 60)
    for name in models:
        if name not in d:
            continue
        p_hat = np.concatenate([np.asarray(p) for p in d[name]["probs"]])
        r = captured_mi(p_hat, y, 40)
        # stability across bin counts (binning MI is bin-count sensitive)
        I_bins = [round(captured_mi(p_hat, y, nb)["I"], 4) for nb in (20, 40, 80)]
        res[name] = {**r, "I_bins_20_40_80": I_bins}
        print(f"{name:<12} | {r['I']:>15.4f} | {r['efficiency']:>11.1%} | "
              f"{r['H_Y_given_phat']:>10.4f}")

    print()
    if "LogReg" in res and "ForwardGRU" in res and "BiGRU" in res:
        I_lr, I_fwd, I_bi = res["LogReg"]["I"], res["ForwardGRU"]["I"], res["BiGRU"]["I"]
        print("Information decomposition (bits/token):")
        print(f"  linear per-token (LogReg)        : {I_lr:.4f}")
        print(f"  + nonlinearity & causal context  : +{I_fwd - I_lr:.4f}  (ForwardGRU {I_fwd:.4f})")
        print(f"  + future/bidirectional context   : +{I_bi - I_fwd:.4f}  (BiGRU {I_bi:.4f})")
        print("  NOTE: the LogReg->ForwardGRU jump mixes NONLINEAR capacity with causal")
        print("  context; server_fano.py separates them (a nonlinear per-token model,")
        print("  HistGBM, reaches 0.0355 bits, so causal context alone is only ~+0.005).")
        print(f"  residual H(Y|p_hat) at BiGRU: {res['BiGRU']['H_Y_given_phat']:.4f} "
              f"({1 - res['BiGRU']['efficiency']:.0%} of label entropy unexplained)")
        print()
        print("Reading: BiGRU (the ceiling architecture) captures only "
              f"{(I_bi/I_fwd - 1)*100:.0f}% more than the causal ForwardGRU,")
        print("and both leave most of H(Y) unexplained. If extra capacity cannot")
        print("extract more, the bottleneck is the features, not the model --")
        print("the information-theoretic reading of the architecture-independent ceiling.")
        print("\nNOTE: I(p_hat;Y) lower-bounds the feature information I(X;Y). A")
        print("definitive 'near the limit' (Bayes error / Fano) needs the raw 33-dim")
        print("features and is a server experiment.")

    json.dump({"pi": pi, "H_Y": float(hb(pi)), "models": res}, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
