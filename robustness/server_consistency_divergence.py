#!/usr/bin/env python3
"""Does the black-box consistency block raise the feature divergence D(P1||P0)?

This is the cheap, no-training half of the consistency experiment (step A). It
answers Noah's question with a number, before any detector is trained.

By the KL chain rule, the increment

    Delta D = D(P1||P0 | base+consistency) - D(P1||P0 | base)
            = E_{P1}[ D( P1(consistency | base) || P0(consistency | base) ) ]

is exactly the CONDITIONAL divergence the consistency block carries on top of the
existing 33-d Text+NLI+LM stream. If Delta D > 0 (beyond noise), consistency is a
genuinely new measurement and the Lorden floor ln(gamma)/D drops; if Delta D ~ 0,
consistency is redundant with the aleatoric LM block on RAGTruth.

CONTROL: D_base is recomputed on the SAME subset of documents for which we have
consistency features (the open-weight generator subset), so Delta D is not
confounded by the subset restriction. We never compare the subset's D to the
full-data 3.5 nats.

Reuses the estimators from server_knn_divergence.py (no reimplementation).
Run on the GPU server in ~/hallucination_exp after server_consistency_features.py.

Usage:
  python server_consistency_divergence.py --consistency consistency_feats_train.json
"""
import argparse
import json

import numpy as np

from run_extended import load_and_enrich_all, assemble_features
from server_knn_divergence import knn_kl, diagonal_gaussian_kl, GAMMA, KS, N_SUB


def build_aligned(consistency_path):
    """Return (X_base, C, y) restricted to docs that have consistency features,
    aligned row-for-row. X_base is the 33-d stream; C is the consistency block."""
    payload = json.load(open(consistency_path))
    feats = payload["feats"] if "feats" in payload else payload
    meta = payload.get("meta", {})

    data = load_and_enrich_all()
    assert data["has_all"], "need all signals (Text+NLI+LM) for the 33-d base"
    base = assemble_features(data["tr_base"], data["tr_nli"], data["tr_lm"], True, True)
    labs = data["tr_labs"]
    ex = data["tr_ex"]

    xb, cc, yy = [], [], []
    matched, mismatched, missing = 0, 0, 0
    for i, e in enumerate(ex):
        cid = str(e["id"])
        if cid not in feats:
            missing += 1
            continue
        cdoc = np.asarray(feats[cid], dtype=np.float64)
        if cdoc.ndim != 2 or cdoc.shape[0] != base[i].shape[0]:
            mismatched += 1
            continue
        xb.append(np.asarray(base[i], dtype=np.float64))
        cc.append(cdoc)
        yy.append(np.asarray(labs[i]))
        matched += 1

    if not matched:
        raise SystemExit("no documents aligned; check ids / token counts")
    print(f"Aligned {matched} docs (missing {missing}, token-count mismatch {mismatched})",
          flush=True)
    X_base = np.vstack(xb)
    C = np.vstack(cc)
    y = np.concatenate(yy)
    return X_base, C, y, meta


def divergences(p1, p0, label):
    d_diag = diagonal_gaussian_kl(p1, p0)
    print(f"\n[{label}] dim={p1.shape[1]}  diag-Gaussian D = {d_diag:.3f} nats "
          f"-> floor = {np.log(GAMMA) / d_diag:.2f} tok")
    knn = {}
    for k in KS:
        d = knn_kl(p1, p0, k)
        floor = float(np.log(GAMMA) / d) if d > 0 else float("inf")
        knn[str(k)] = {"D": d, "floor": floor}
        print(f"    k={k:2d}: D = {d:7.3f} nats -> floor = {floor:.2f} tok")
    med = float(np.median([knn[str(k)]["D"] for k in KS]))
    return {"d_diag": d_diag, "knn": knn, "knn_median_D": med}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consistency", default="consistency_feats_train.json")
    ap.add_argument("--out", default="consistency_divergence.json")
    args = ap.parse_args()

    X_base, C, y = build_aligned(args.consistency)[:3]
    X_aug = np.hstack([X_base, C])
    # negative control: same number of PURE-NOISE dims. If Delta D_kNN for consistency
    # is just a curse-of-dimensionality artifact (more dims inflate distances), the
    # random control shows the same lift. Consistency is real only if it beats this.
    rngc = np.random.RandomState(1)
    X_rand = np.hstack([X_base, rngc.standard_normal((X_base.shape[0], C.shape[1]))])
    print(f"\nFeatures: base={X_base.shape[1]}d, +consistency={X_aug.shape[1]}d; "
          f"clean {int((y < 0.5).sum()):,} / hallu {int((y > 0.5).sum()):,} tokens",
          flush=True)

    # z-score per column so the k-NN Euclidean metric is scale-fair: the [0,1]
    # consistency dims would be nearly invisible next to large-scale base dims in
    # raw Euclidean distance, biasing Delta D_kNN toward 0. The diagonal-Gaussian KL
    # is invariant to per-dim scaling, so this leaves those numbers unchanged and
    # only makes the k-NN estimate meaningful.
    def _z(X):
        return (X - X.mean(0)) / (X.std(0) + 1e-8)

    X_base = _z(X_base)
    X_aug = _z(X_aug)
    X_rand = _z(X_rand)

    rng = np.random.RandomState(0)

    def split_and_sub(X):
        clean = X[y < 0.5]
        hallu = X[y > 0.5]
        # same subsample indices for base and aug so Delta D is paired
        idx0 = rng.permutation(len(clean))[:min(N_SUB, len(clean))]
        idx1 = rng.permutation(len(hallu))[:min(N_SUB, len(hallu))]
        return hallu[idx1], clean[idx0]

    # paired subsampling: reset rng so base and aug pick the SAME rows
    rng = np.random.RandomState(0)
    p1_base, p0_base = split_and_sub(X_base)
    rng = np.random.RandomState(0)
    p1_aug, p0_aug = split_and_sub(X_aug)
    rng = np.random.RandomState(0)
    p1_rand, p0_rand = split_and_sub(X_rand)

    res = {}
    res["n_base_dim"] = int(X_base.shape[1])
    res["n_aug_dim"] = int(X_aug.shape[1])
    res["base"] = divergences(p1_base, p0_base, "base (33-d, subset)")
    res["aug"] = divergences(p1_aug, p0_aug, "base + consistency")
    res["rand_control"] = divergences(p1_rand, p0_rand, "base + 2 random dims (control)")

    dD_diag = res["aug"]["d_diag"] - res["base"]["d_diag"]
    dD_knn = res["aug"]["knn_median_D"] - res["base"]["knn_median_D"]
    dD_rand = res["rand_control"]["knn_median_D"] - res["base"]["knn_median_D"]
    res["delta_D_diag"] = dD_diag
    res["delta_D_knn_median"] = dD_knn
    res["delta_D_rand_knn_median"] = dD_rand
    print(f"\n=== Delta D (conditional divergence of consistency | 33-d) ===")
    print(f"  diag-Gaussian        : {dD_diag:+.3f} nats")
    print(f"  k-NN (median)        : {dD_knn:+.3f} nats")
    print(f"  k-NN random control  : {dD_rand:+.3f} nats   (consistency real iff dD_kNN >> this)")
    print(f"  floor (k-NN median): {np.log(GAMMA)/res['base']['knn_median_D']:.2f}"
          f" -> {np.log(GAMMA)/res['aug']['knn_median_D']:.2f} tok"
          if res["aug"]["knn_median_D"] > 0 else "")
    print("\nReading: Delta D > 0 beyond noise => consistency is a new measurement, "
          "floor drops. Delta D ~ 0 => redundant with the LM block.")

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
