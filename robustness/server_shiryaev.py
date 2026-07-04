#!/usr/bin/env python3
"""Direction 3: Shiryaev-Roberts and the Bayesian Shiryaev filter vs CUSUM.

CUSUM is minimax -- optimal against the worst-case change time. But we KNOW the onset
hazard, p = P(onset at t | faithful at t-1) ~ 0.0044 (Assumption 1 / appendix A). A
detector that uses it should do at least as well. Two classical alternatives, run on
the SAME learned increment Y_t = logit(p_hat_t) - k as the ForwardGRU-CUSUM:

  Shiryaev-Roberts : R_t = (1 + R_{t-1}) * LR_t,  LR_t = exp(Y_t), threshold on R_t.
                     Sums the likelihood ratio over all putative change points (an
                     improper uniform prior on the onset); often beats CUSUM slightly.
  Shiryaev filter  : the exact Bayesian change posterior under a geometric(p) prior,
                     pi_t = P(theta <= t | x_{1:t}), thresholded. This is the
                     finite-dimensional optimal detector of Corollary 1, using p.

We compare censored EDD and recall at matched ARL0 against the CUSUM baseline. Runs on
the saved ForwardGRU posteriors; pure numpy + the eval machinery. No GPU.
"""
import json
from pathlib import Path

import numpy as np

from run_learned_cusum import (
    cusum_reference_value, cusum_path, first_onset, sweep, at_arl0,
)

HERE = Path(__file__).resolve().parent
PROBS = HERE / "directional_probs_seed42.json"
GAMMA_TARGETS = [50, 100, 200]
ARL0_MAIN = 100
P_HAZARD = 0.0044     # onset hazard from the label Markov chain (appendix A)


def increments(probs, ref):
    p = np.clip(np.asarray(probs, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)) - ref


def sr_path(probs, ref):
    """Shiryaev-Roberts in the log domain: L_t = Y_t + softplus(L_{t-1})."""
    Y = increments(probs, ref)
    L = -np.inf
    out = np.empty(len(Y))
    for t in range(len(Y)):
        sp = np.logaddexp(0.0, L) if np.isfinite(L) else 0.0   # log(1 + e^{L})
        L = Y[t] + sp
        out[t] = L
    return out


def shiryaev_path(probs, ref, p):
    """Exact Bayesian change posterior pi_t under a geometric(p) onset prior."""
    Y = increments(probs, ref)
    LR = np.exp(np.clip(Y, -30, 30))
    pi = 0.0
    out = np.empty(len(Y))
    for t in range(len(Y)):
        pred = pi + (1 - pi) * p          # prior-predict the change by t
        num = pred * LR[t]
        pi = num / (num + (1 - pred))      # Bayes update
        out[t] = pi
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
        print(f"  {name:18s} ARL0=100: censored EDD={op['edd']:.1f}  "
              f"delay={op['delay']:.1f}  recall={op['recall']:.2f}  @ARL0={op['arl0']:.0f}",
              flush=True)
    return res


def main():
    data = json.load(open(PROBS))
    fwd = data["ForwardGRU"]
    labs = [np.asarray(l, float) for l in fwd["labs"]]
    onsets = [first_onset(l) for l in labs]
    ref, mu0, mu1 = cusum_reference_value(fwd["probs"], fwd["labs"])
    print(f"Docs {len(labs)}, hallu {sum(o is not None for o in onsets)}; "
          f"ref={ref:.3f}, hazard p={P_HAZARD}", flush=True)

    cusum = [cusum_path(p, ref) for p in fwd["probs"]]
    sr = [sr_path(p, ref) for p in fwd["probs"]]
    shir = [shiryaev_path(p, ref, P_HAZARD) for p in fwd["probs"]]

    print("\nAt matched ARL0 (censored EDD / delay-among-detected / recall):")
    out = {}
    out["CUSUM"] = evaluate("CUSUM (baseline)", cusum, onsets, np.linspace(0, 120, 801))
    out["Shiryaev-Roberts"] = evaluate("Shiryaev-Roberts", sr, onsets, np.linspace(-20, 300, 1201))
    out["Shiryaev"] = evaluate("Shiryaev(p)", shir, onsets,
                               np.unique(np.concatenate([np.linspace(0, 0.999, 600),
                                                         1 - np.logspace(-6, -1, 300)])))
    print("\nReading: Shiryaev/SR below CUSUM => exploiting the known hazard helps; "
          "equal => CUSUM's minimax rule already captures the structure.")
    json.dump(out, open(HERE / "shiryaev.json", "w"), indent=2)
    print(f"Saved -> {HERE/'shiryaev.json'}")


if __name__ == "__main__":
    main()
