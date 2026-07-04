#!/usr/bin/env python3
"""Why does the +0.83 nats of raw consistency divergence (H1) not help a trained
detector (H2: token-AUC and EDD flat within seed noise)?

Two hypotheses:
  (1) REDUNDANCY  -- consistency is (nonlinearly) predictable from the 33-d base,
      so a nonlinear GRU already implies it: high R^2 of base -> consistency.
  (2) NON-EXTRACTABILITY -- consistency is NOT predictable from base, yet the model
      fails to use it: low R^2 but H2 still flat.

This is a CPU-only diagnostic (no GPU): fit a nonlinear regressor base(33-d) ->
each consistency dim on the train split and report held-out R^2. Reuses the exact
aligned subset and split from server_consistency_h2 so the numbers line up with H2.

Reading: R^2 >~ 0.5 on a consistency dim => that dim is largely redundant with the
base signal (explains the H2 null). R^2 ~ 0 => not redundant => non-extractability.
"""
import argparse
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from server_consistency_h2 import build_subset, split_indices


def stack(F, labs, idx):
    X = np.vstack([np.asarray(F[j]) for j in idx]).astype(np.float64)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consistency", default="consistency_feats_train_llama7b.json")
    ap.add_argument("--out", default="consistency_redundancy.json")
    args = ap.parse_args()

    base, aug, labs, ex = build_subset(args.consistency)
    n_cons = aug[0].shape[1] - base[0].shape[1]
    tr_i, val_i, te_i = split_indices(len(base))

    # token matrices on the H2 split; consistency dims are the trailing columns of aug
    Xtr = np.vstack([np.asarray(base[j]) for j in tr_i]).astype(np.float64)
    Xte = np.vstack([np.asarray(base[j]) for j in te_i]).astype(np.float64)
    Ctr = np.vstack([np.asarray(aug[j])[:, -n_cons:] for j in tr_i]).astype(np.float64)
    Cte = np.vstack([np.asarray(aug[j])[:, -n_cons:] for j in te_i]).astype(np.float64)
    print(f"base->consistency regression: train {Xtr.shape}, test {Xte.shape}, "
          f"{n_cons} consistency dims", flush=True)

    dim_names = ["selfcheck_nli", "selfcheck_ngram"][:n_cons]
    res = {"dims": {}}
    for d in range(n_cons):
        name = dim_names[d] if d < len(dim_names) else f"dim{d}"
        ytr, yte = Ctr[:, d], Cte[:, d]
        lin = LinearRegression().fit(Xtr, ytr)
        r2_lin = float(r2_score(yte, lin.predict(Xte)))
        gb = HistGradientBoostingRegressor(max_iter=300, max_depth=4,
                                           learning_rate=0.08).fit(Xtr, ytr)
        r2_gb = float(r2_score(yte, gb.predict(Xte)))
        res["dims"][name] = {"r2_linear": r2_lin, "r2_nonlinear": r2_gb,
                             "var_test": float(np.var(yte))}
        print(f"  {name:16s}: R^2 linear={r2_lin:+.3f}  nonlinear(GBT)={r2_gb:+.3f}  "
              f"(test var={np.var(yte):.4f})", flush=True)

    print("\nReading: high nonlinear R^2 => consistency is implied by the 33-d base "
          "(redundant) -> explains why H2 is flat. Low R^2 => not redundant "
          "(non-extractability / objective mismatch).")
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
