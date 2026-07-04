#!/usr/bin/env python3
"""PoC for direction B: what actually limits the realized rate I, and is the
pre-change score noise sigma_0 a real lever?

Closed form (centered-reference CUSUM, Gaussian increments, Wolfram-derived):
    I = omega * E1[Y] = 2 m^2 / sigma0^2,    EDD ~ ln(gamma) * sigma0^2 / (2 m^2),
where Y = logit(p_hat) - k is the per-token score increment, m = (E1[Y]-E0[Y])/2
is the half mean-gap (~ token separability), sigma0 = std of Y on CLEAN tokens.
Note I does NOT depend on sigma1 (post-change spread) -- so the levers are m and
sigma0, NOT the naive "symmetrize the post-change variance".

This script measures m, sigma0, sigma1 and the non-Gaussianity of the learned
score increments from the SAVED ForwardGRU/BiGRU/LogReg probs (no GPU, no retrain),
checks whether the Gaussian closed form reproduces the empirical realized rate, and
reports the leverage of each knob. Decisive question: is the deficit binding on
sigma0 (noisy clean baseline) or on m (weak hallucinated firing)?
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import skew, kurtosis

HERE = Path(__file__).resolve().parent
PROBS = HERE / "directional_probs_seed42.json"
GAMMA = 100.0
D_FEATURE = 2.8   # feature-space divergence (nats), sets the Lorden floor


def increments(probs, labs):
    p = np.clip(np.concatenate([np.asarray(x, float) for x in probs]), 1e-6, 1 - 1e-6)
    y = np.concatenate([np.asarray(x, float) for x in labs])
    lg = np.log(p / (1 - p))                       # logits
    mu0, mu1 = lg[y < 0.5].mean(), lg[y > 0.5].mean()
    k = 0.5 * (mu0 + mu1)                           # centered CUSUM reference
    Y = lg - k
    return Y[y < 0.5], Y[y > 0.5]                   # pre-change (clean), post-change (hallu)


def lundberg_omega(Y0):
    """omega>0 solving E0[exp(omega Y)] = 1."""
    f = lambda w: float(np.mean(np.exp(w * Y0)) - 1.0)
    hi = 0.5
    while f(hi) < 0 and hi < 100:
        hi *= 1.5
    return brentq(f, 1e-9, hi)


def gauss_kl(m1, s1, m0, s0):
    """KL(N(m1,s1^2) || N(m0,s0^2)) in nats."""
    return float(np.log(s0 / s1) + (s1**2 + (m1 - m0) ** 2) / (2 * s0**2) - 0.5)


def analyze(name, probs, labs):
    Y0, Y1 = increments(probs, labs)
    mu0, mu1 = Y0.mean(), Y1.mean()
    s0, s1 = Y0.std(), Y1.std()
    m = 0.5 * (mu1 - mu0)                            # half mean-gap
    rho = s1 / s0
    omega = lundberg_omega(Y0)
    I_emp = float(omega * mu1)                       # realized rate (E1[Y]=mu1)
    omega_g = 2 * m / s0**2
    I_g = 2 * m**2 / s0**2                           # Gaussian closed form
    edd_emp = GAMMA and np.log(GAMMA) / I_emp if I_emp > 0 else np.inf
    d_inc = gauss_kl(mu1, s1, mu0, s0)               # increment-space divergence (Gaussian approx)

    print(f"\n=== {name} ===")
    print(f"  mean-gap: E0[Y]={mu0:+.3f}  E1[Y]={mu1:+.3f}  m={m:.3f}")
    print(f"  spread:   sigma0(clean)={s0:.3f}  sigma1(hallu)={s1:.3f}  rho=sigma1/sigma0={rho:.2f}")
    print(f"  shape:    skew0={skew(Y0):+.2f} kurt0={kurtosis(Y0):+.2f} | "
          f"skew1={skew(Y1):+.2f} kurt1={kurtosis(Y1):+.2f}")
    print(f"  rate:     omega_emp={omega:.3f}  I_emp={I_emp:.3f} | "
          f"omega_gauss={omega_g:.3f}  I_gauss={I_g:.3f}  (I_emp/I_gauss={I_emp/I_g:.2f})")
    print(f"  EDD@{int(GAMMA)}:  {edd_emp:.1f} tok   (closed form ln(g)*s0^2/(2m^2)={np.log(GAMMA)*s0**2/(2*m**2):.1f})")
    print(f"  deficit:  D_feature/I={D_FEATURE/I_emp:.1f}x  =  "
          f"(D_feat/D_inc={D_FEATURE/d_inc:.1f}) x (D_inc/I={d_inc/I_emp:.1f})")
    # leverage: EDD ~ s0^2 / m^2.  -20% sigma0  vs  +20% m
    edd_lo_s0 = np.log(GAMMA) * (0.8 * s0) ** 2 / (2 * m**2)
    edd_hi_m = np.log(GAMMA) * s0**2 / (2 * (1.2 * m) ** 2)
    print(f"  lever:    EDD if sigma0 -20% -> {edd_lo_s0:.1f} tok ;  if m +20% -> {edd_hi_m:.1f} tok "
          f"(both elasticity 2; which has headroom?)")
    return {"name": name, "m": m, "sigma0": s0, "sigma1": s1, "rho": rho,
            "skew0": float(skew(Y0)), "kurt0": float(kurtosis(Y0)),
            "skew1": float(skew(Y1)), "kurt1": float(kurtosis(Y1)),
            "omega_emp": omega, "I_emp": I_emp, "I_gauss": I_g,
            "edd_emp": float(edd_emp), "d_increment": d_inc,
            "deficit_feature": D_FEATURE / I_emp, "deficit_increment": d_inc / I_emp}


def main():
    data = json.load(open(PROBS))
    out = {}
    for name in ["LogReg", "ForwardGRU", "BiGRU"]:
        if name in data:
            out[name] = analyze(name, data[name]["probs"], data[name]["labs"])
    print("\nReading: I = 2 m^2 / sigma0^2 -- sigma1 does NOT enter. The deployable lever "
          "for delay is a QUIETER clean baseline (sigma0 down) or stronger hallu firing (m up). "
          "If I_emp << I_gauss, non-Gaussian tails (kurt0) inflate omega's shortfall -> a "
          "tail-robust score is the lever instead.")
    json.dump(out, open(HERE / "symmetry_poc.json", "w"), indent=2)
    print(f"\nSaved -> {HERE/'symmetry_poc.json'}")


if __name__ == "__main__":
    main()
