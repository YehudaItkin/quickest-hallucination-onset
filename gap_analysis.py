#!/usr/bin/env python3
"""Closing-the-gap analysis: what moves the realized information rate
I(g) = omega * E_1[Y] of the learned score toward the feature divergence D,
and what the i.i.d. first-order rate misses.

Three tests, all local on the saved causal ForwardGRU posteriors:
  A. Temperature scaling s -> s/T. Predicted INVARIANT: omega -> omega*T,
     delta1 -> delta1/T, so I = omega*delta1 is unchanged. If so, the 4.5x
     information-rate shortfall is about the SHAPE of the score, not its scale,
     and "recalibrate by temperature" cannot close it.
  B. Isotonic (monotone nonlinear) recalibration of s to the labels. A nonlinear
     monotone map can change I. Tests whether reshaping the score helps.
  C. Integrated autocorrelation time tau of the score on the clean stream. The
     i.i.d. rate treats increments as independent; if they are correlated, each
     token carries ~1/tau the independent evidence, inflating EDD by ~tau. If
     tau ~ 1.9 it accounts for the residual factor in the gap decomposition.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
PROBS = HERE / "directional_probs_seed42.json"
D_FEATURE = 3.5   # diagonal-Gaussian feature KL (nats), POC stage 3
OUT = HERE / "gap_analysis.json"


def load():
    d = json.load(open(PROBS))["ForwardGRU"]
    probs = [np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6) for p in d["probs"]]
    labs = [np.asarray(l, dtype=np.float64) for l in d["labs"]]
    return probs, labs


def logit(p):
    return np.log(p / (1 - p))


def info_rate(s_clean, s_hallu):
    """I = omega * E_1[Y], Y = s - k, k = (mu0+mu1)/2 (Siegmund first-order)."""
    mu0, mu1 = float(s_clean.mean()), float(s_hallu.mean())
    k = 0.5 * (mu0 + mu1)
    Yc, Yh = s_clean - k, s_hallu - k
    delta1 = float(Yh.mean())

    def mgf(w):
        return float(np.mean(np.exp(w * Yc)) - 1.0)

    hi = 0.5
    while mgf(hi) < 0 and hi < 80:
        hi *= 1.5
    omega = float(brentq(mgf, 1e-9, hi))  # type: ignore[arg-type]
    return {"mu0": mu0, "mu1": mu1, "omega": omega,
            "delta1": delta1, "I": omega * delta1}


def integrated_autocorr_time(seqs, max_lag=15):
    """tau = 1 + 2 sum_k rho_k, pooled within-sequence autocorrelation of the
    score, truncated at the first non-positive rho_k. seqs = list of 1-D arrays
    (per-document score sequences from one regime)."""
    seqs = [s for s in seqs if len(s) > max_lag + 1]
    allv = np.concatenate(seqs)
    mu, var = allv.mean(), allv.var()
    rhos = []
    for k in range(1, max_lag + 1):
        num, n = 0.0, 0
        for s in seqs:
            a, b = s[:-k] - mu, s[k:] - mu
            num += float(np.sum(a * b))
            n += len(a)
        rho = (num / n) / var if n > 0 and var > 0 else 0.0
        if rho <= 0:
            break
        rhos.append(rho)
    tau = 1.0 + 2.0 * sum(rhos)
    return tau, rhos


def main():
    probs, labs = load()
    s_all = [logit(p) for p in probs]
    clean = np.concatenate([s[l < 0.5] for s, l in zip(s_all, labs)])
    hallu = np.concatenate([s[l > 0.5] for s, l in zip(s_all, labs)])

    base = info_rate(clean, hallu)
    print(f"Baseline: I(g_hat)={base['I']:.3f} nats  (omega={base['omega']:.3f}, "
          f"delta1={base['delta1']:.3f})  ->  D/I = {D_FEATURE/base['I']:.2f}x")
    print(f"  predicted EDD@100 = ln(100)/I = {np.log(100)/base['I']:.2f} tok\n")

    # --- Test A: temperature scaling (expect invariance) ---
    print("Test A: temperature scaling s -> s/T")
    tempA = {}
    for T in [0.5, 2.0, 4.0]:
        r = info_rate(clean / T, hallu / T)
        tempA[T] = r["I"]
        print(f"  T={T:<4}: I={r['I']:.3f}  (omega={r['omega']:.3f}, delta1={r['delta1']:.3f})")
    print("  -> I invariant to scale: the shortfall is the score's SHAPE, not its scale\n")

    # --- Test B: isotonic (monotone nonlinear) recalibration ---
    print("Test B: isotonic recalibration of the posterior to labels")
    from sklearn.isotonic import IsotonicRegression
    p_all = np.concatenate([p for p in probs])
    y_all = np.concatenate([l for l in labs])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(p_all, y_all)
    s_iso = [logit(np.clip(iso.transform(p), 1e-6, 1 - 1e-6)) for p in probs]
    clean_i = np.concatenate([s[l < 0.5] for s, l in zip(s_iso, labs)])
    hallu_i = np.concatenate([s[l > 0.5] for s, l in zip(s_iso, labs)])
    rB = info_rate(clean_i, hallu_i)
    print(f"  isotonic I={rB['I']:.3f}  (omega={rB['omega']:.3f}, delta1={rB['delta1']:.3f})  "
          f"->  D/I = {D_FEATURE/rB['I']:.2f}x")
    print(f"  change vs baseline: {100*(rB['I']/base['I']-1):+.1f}%\n")

    # --- Test C: autocorrelation of the score on the clean stream ---
    print("Test C: integrated autocorrelation time of the score (clean stream)")
    clean_docs = [s for s, l in zip(s_all, labs) if not (l > 0.5).any()]
    tau, rhos = integrated_autocorr_time(clean_docs)
    print(f"  rho_1..rho_k = {[round(r,3) for r in rhos]}")
    print(f"  integrated autocorr time tau = {tau:.2f}  (score is strongly smoothed)\n")

    # --- Test D: process-level adjustment coefficient from the ARL0-h curve ---
    # The marginal MGF gives omega; the true correlation-aware omega* is the slope
    # of ln(ARL0) vs threshold h on the detector's own clean operating curve.
    print("Test D: correlation-aware adjustment coefficient omega* (from ARL0-h curve)")
    omega_star = None
    cu = HERE / "learned_cusum.json"
    if cu.exists():
        curve = json.load(open(cu))["curves"]["ForwardGRU-CUSUM"]
        hs = np.array([r["threshold"] for r in curve])
        arl = np.array([r["arl0"] for r in curve])
        m = (~np.isnan(arl)) & (arl > 130) & (arl < 1000) & (hs > 1)
        if m.sum() >= 5:
            omega_star = float(np.polyfit(hs[m], np.log(arl[m]), 1)[0])
            I_corr = omega_star * base["delta1"]
            print(f"  omega* = d ln(ARL0)/dh = {omega_star:.4f}  (marginal omega={base['omega']:.3f}, "
                  f"ratio {base['omega']/omega_star:.1f}x ~ tau={tau:.0f})")
            print(f"  asymptotic I_corr = omega*·delta1 = {I_corr:.4f} -> "
                  f"EDD@100 = {np.log(100)/I_corr:.0f} tok  (OVERSHOOTS observed ~11.5)")
            print(f"  => asymptotic dependent-data rate does NOT tighten the prediction:")
            print(f"     detection (~11 tok) is faster than mixing (tau~{tau:.0f}); the reset")
            print(f"     CUSUM also floors ARL0 at the mean clean-document length (~116).\n")

    # --- honest gap summary ---
    floor = np.log(100) / D_FEATURE
    iid_pred = np.log(100) / base["I"]
    print("Gap at ARL0=100 (honest):")
    print(f"  floor (D={D_FEATURE} nats)              = {floor:.2f} tok")
    print(f"  i.i.d. rate with learned score      = {iid_pred:.2f} tok  "
          f"(shape factor D/I = {D_FEATURE/base['I']:.1f}x; isotonic recovers only +{100*(rB['I']/base['I']-1):.0f}%)")
    print(f"  observed ForwardGRU-CUSUM           ~ 11.5 tok  "
          f"(residual {11.5/iid_pred:.1f}x: finite-horizon, NOT closed by asymptotic correlation)")

    json.dump({
        "D_feature": D_FEATURE, "baseline": base,
        "temperature_invariance": {str(t): v for t, v in tempA.items()},
        "isotonic": rB, "autocorr_rhos": rhos, "tau_int": tau,
        "omega_star": omega_star,
        "floor": float(floor), "iid_predicted_edd_100": float(iid_pred),
    }, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
