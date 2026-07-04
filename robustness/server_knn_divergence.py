#!/usr/bin/env python3
"""Robustness check for the Lorden floor: estimate the feature divergence
D(P_1 || P_0) with a NON-parametric k-NN estimator and compare it to the
diagonal-Gaussian estimate (D ~ 3.5 nats) the paper uses.

Motivation (reviewer concern): the floor edd_min = ln(gamma)/D and every
"x above the floor" ratio rest on a diagonal-Gaussian KL, yet the paper itself
argues a diagonal Gaussian is the WRONG model for the 33-d feature law (it is why
the parametric CUSUM fails). This script checks whether the floor is an artifact
of that estimator by re-estimating D with the Wang-Kulkarni-Verdu (2009) k-NN
divergence estimator, which makes no parametric assumption.

Estimator (nats), for X_i ~ P_1 (hallucinated tokens), Y_j ~ P_0 (faithful):
    D_hat(P1||P0) = (d/n) sum_i ln( nu_k(i) / rho_k(i) ) + ln( m / (n-1) )
  rho_k(i): distance from X_i to its k-th NN among the OTHER X (P_1) points;
  nu_k(i):  distance from X_i to its k-th NN among ALL Y (P_0) points.

Run on the GPU server in ~/hallucination_exp (only needs the feature loader).
"""
import json

import numpy as np
from sklearn.neighbors import NearestNeighbors

from run_extended import load_and_enrich_all, assemble_features

RNG = np.random.RandomState(0)
N_SUB = 40000          # subsample per class for a tractable k-NN search
KS = [3, 5, 10]        # k for the k-NN estimator (report several for stability)
GAMMA = 100            # ARL0 operating point (alpha = 0.01)


def diagonal_gaussian_kl(p1, p0):
    """D(P1||P0) under independent per-dim Gaussians, in nats (the paper's 3.5)."""
    m0, s0 = p0.mean(0), p0.std(0) + 1e-8
    m1, s1 = p1.mean(0), p1.std(0) + 1e-8
    return float(0.5 * np.sum(np.log(s0**2 / s1**2)
                              + (s1**2 + (m1 - m0)**2) / s0**2 - 1.0))


def knn_kl(p1, p0, k):
    """Wang-Kulkarni-Verdu k-NN estimate of D(P1||P0) in nats."""
    n, d = p1.shape
    m = p0.shape[0]
    # rho: k-th NN of each P1 point among the OTHER P1 points (k+1 to skip self)
    nn1 = NearestNeighbors(n_neighbors=k + 1).fit(p1)
    rho = nn1.kneighbors(p1)[0][:, k]
    # nu: k-th NN of each P1 point among the P0 points
    nn0 = NearestNeighbors(n_neighbors=k).fit(p0)
    nu = nn0.kneighbors(p1)[0][:, k - 1]
    eps = 1e-12
    rho = np.maximum(rho, eps)
    nu = np.maximum(nu, eps)
    return float((d / n) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1)))


def subsample(x, n):
    if len(x) <= n:
        return x
    idx = RNG.choice(len(x), n, replace=False)
    return x[idx]


def main():
    data = load_and_enrich_all()
    assert data["has_all"], "need all signals"
    tr = assemble_features(data["tr_base"], data["tr_nli"], data["tr_lm"], True, True)
    X = np.vstack(tr).astype(np.float64)
    y = np.concatenate([np.asarray(l) for l in data["tr_labs"]])
    clean = X[y < 0.5]          # P0
    hallu = X[y > 0.5]          # P1
    print(f"Features: {X.shape[1]}-d; clean {len(clean):,} / hallu {len(hallu):,} tokens")

    # raw features (same space as the diagonal-Gaussian D=3.5)
    p0 = subsample(clean, N_SUB)
    p1 = subsample(hallu, N_SUB)

    d_diag = diagonal_gaussian_kl(p1, p0)
    print(f"\nDiagonal-Gaussian D(P1||P0) = {d_diag:.3f} nats "
          f"-> floor ln({GAMMA})/D = {np.log(GAMMA)/d_diag:.2f} tok")

    res = {"d_diag_gauss": d_diag, "floor_diag": float(np.log(GAMMA) / d_diag),
           "knn": {}}
    print("\nNon-parametric k-NN divergence (Wang-Kulkarni-Verdu):")
    for k in KS:
        d_knn = knn_kl(p1, p0, k)
        floor = float(np.log(GAMMA) / d_knn) if d_knn > 0 else float("inf")
        res["knn"][str(k)] = {"D": d_knn, "floor": floor}
        print(f"  k={k:2d}: D = {d_knn:7.3f} nats -> floor = {floor:.2f} tok")

    d_knn_vals = [res["knn"][str(k)]["D"] for k in KS]
    print(f"\nReading: if the k-NN D (median {np.median(d_knn_vals):.2f} nats) is close to the "
          f"diagonal-Gaussian {d_diag:.2f}, the 1.3-token floor is robust to the estimator;")
    print("if it is much larger, the true floor is even smaller and the gap we report is, "
          "if anything, conservative.")
    res["knn_median_D"] = float(np.median(d_knn_vals))
    res["n_sub"] = int(min(N_SUB, len(clean), len(hallu)))
    json.dump(res, open("server_knn_divergence.json", "w"), indent=2)
    print("\nSaved -> server_knn_divergence.json")


if __name__ == "__main__":
    main()
