#!/usr/bin/env python3
"""H2: does the black-box consistency block convert into faster onset detection?

H1 (server_consistency_divergence.py) showed consistency adds ~+0.83 nats of
*non-Gaussian* conditional divergence over the 33-d base (diag-Gaussian sees ~0).
Non-Gaussian => a linear detector cannot use it, but a nonlinear causal recurrent
one (the learned CUSUM) might. H2 is the decider: train ForwardGRU-CUSUM on

    base (33-d)   vs   base + consistency (35-d)

on the SAME llama-2-7b-chat subset and split, and compare
  - realized information rate I(g_hat)  (higher = closer to the Lorden floor)
  - achieved detection delay EDD at matched ARL0=100  (lower = faster)

If aug beats base on a held-out set, the +0.83 nats become real speed -- the
direct answer to Noah's question -- while staying fully black-box.

Reuses train_nn / assemble_features / load_and_enrich_all from run_extended and the
realized-rate + change-point machinery from run_learned_cusum; neither is modified.
ForwardGRU (causal, forward-only -> valid for streaming) is defined here because
run_extended only ships the bidirectional variants.

Run on the GPU server in ~/hallucination_exp after the consistency features exist.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from run_extended import load_and_enrich_all, assemble_features, train_nn, set_seed
from run_learned_cusum import (
    cusum_reference_value, information_rate, cusum_path, first_onset,
    sweep, at_arl0,
)

ARL0_TARGET = 100          # Lorden's alpha=0.01 operating point
CUSUM_GRID = np.linspace(0.0, 120.0, 601)
SPLIT_SEED = 0             # fixed train/val/test split so base and aug are paired


class ForwardGRU(nn.Module):
    """Causal (forward-only) GRU labeler = the learned CUSUM of the paper."""

    def __init__(self, dim, h=64):
        super().__init__()
        self.gru = nn.GRU(dim, h, num_layers=2, batch_first=True,
                          bidirectional=False, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.head(self.gru(x)[0]).squeeze(-1)


def diag_gaussian_D(feats, labs):
    """Diagonal-Gaussian D(P1||P0) on a feature set, for the D/I gap factor."""
    X = np.vstack([np.asarray(f) for f in feats]).astype(np.float64)
    y = np.concatenate([np.asarray(l) for l in labs])
    p0, p1 = X[y < 0.5], X[y > 0.5]
    m0, s0 = p0.mean(0), p0.std(0) + 1e-8
    m1, s1 = p1.mean(0), p1.std(0) + 1e-8
    return float(0.5 * np.sum(np.log(s0**2 / s1**2)
                              + (s1**2 + (m1 - m0)**2) / s0**2 - 1.0))


def build_subset(consistency_path):
    """base + aug (base+consistency) feature lists on the consistency subset,
    aligned by example id; plus labels and examples in one fixed order."""
    payload = json.load(open(consistency_path))
    feats = payload["feats"] if "feats" in payload else payload

    data = load_and_enrich_all()
    assert data["has_all"], "need Text+NLI+LM for the 33-d base"
    base_all = assemble_features(data["tr_base"], data["tr_nli"], data["tr_lm"], True, True)
    labs_all, ex_all = data["tr_labs"], data["tr_ex"]

    base, aug, labs, ex = [], [], [], []
    skipped = 0
    for i, e in enumerate(ex_all):
        cid = str(e["id"])
        if cid not in feats:
            continue
        c = np.asarray(feats[cid], dtype=np.float32)
        b = np.asarray(base_all[i], dtype=np.float32)
        if c.ndim != 2 or c.shape[0] != b.shape[0]:
            skipped += 1
            continue
        base.append(b)
        aug.append(np.hstack([b, c]).astype(np.float32))
        labs.append(np.asarray(labs_all[i], dtype=np.float32))
        ex.append(e)
    print(f"Subset: {len(base)} docs (token-count mismatch skipped {skipped})", flush=True)
    return base, aug, labs, ex


def split_indices(n, seed=SPLIT_SEED, frac=(0.70, 0.15)):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    ntr = int(frac[0] * n)
    nval = int(frac[1] * n)
    return perm[:ntr], perm[ntr:ntr + nval], perm[ntr + nval:]


def evaluate_probs(te_probs, te_labs, D_feature):
    """Realized rate + ARL0=100 operating point from per-doc test probs.

    Also returns the two quantities that separate the "divergence up / timing flat"
    story from a dull "noise dims hurt the model": token-AUC (static separability the
    model actually extracts) and drift0 = E_0[Y] (pre-change drift -- the contamination
    the piecewise-constant argument predicts). Mechanism prediction for aug vs base:
    token_auc up-or-flat, drift0 up (less negative) => omega down => I down => EDD up.
    """
    ref, _, _ = cusum_reference_value(te_probs, te_labs)
    irate = information_rate(te_probs, te_labs, ref)
    onsets = [first_onset(l) for l in te_labs]
    clean_idx = [i for i, o in enumerate(onsets) if o is None]
    hallu_idx = [i for i, o in enumerate(onsets) if o is not None]
    paths = [cusum_path(p, ref) for p in te_probs]
    rows = sweep([paths[i] for i in clean_idx],
                 [paths[i] for i in hallu_idx],
                 [onsets[i] for i in hallu_idx], CUSUM_GRID)
    op = at_arl0(rows, ARL0_TARGET)
    I = irate["I"]
    all_p = np.concatenate([np.asarray(p) for p in te_probs])
    all_y = np.concatenate([np.asarray(l) for l in te_labs])
    token_auc = (float(roc_auc_score(all_y, all_p))
                 if len(np.unique(all_y)) > 1 else float("nan"))
    return {
        "I": I, "omega": irate["omega"], "delta1": irate["delta1"],
        "drift0": irate["drift0"],
        "token_auc": token_auc,
        "D_over_I": (D_feature / I) if I > 0 else float("inf"),
        "edd": op["delay_edd"] if op else float("nan"),
        "delay_detected": op["delay"] if op else float("nan"),
        "recall": op["recall"] if op else float("nan"),
        "arl0": op["arl0"] if op else float("nan"),
        "n_hallu_test": len(hallu_idx),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consistency", default="consistency_feats_train_llama7b.json")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="consistency_h2.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    base, aug, labs, ex = build_subset(args.consistency)
    tr_i, val_i, te_i = split_indices(len(base))
    print(f"Split: train {len(tr_i)} / val {len(val_i)} / test {len(te_i)}", flush=True)

    def sub(F, ii):
        return [F[j] for j in ii]

    D_base = diag_gaussian_D(base, labs)
    D_aug = diag_gaussian_D(aug, labs)
    print(f"diag-Gaussian D: base={D_base:.3f}  aug={D_aug:.3f} nats", flush=True)

    feature_sets = {"base": (base, D_base), "aug": (aug, D_aug)}
    results = {"base": [], "aug": []}
    for seed in args.seeds:
        for name, (F, D) in feature_sets.items():
            set_seed(seed)
            model = ForwardGRU(F[0].shape[1]).to(device)
            te_probs, te_labs = train_nn(
                model, sub(F, tr_i), sub(labs, tr_i), sub(F, val_i), sub(labs, val_i),
                sub(F, te_i), sub(labs, te_i), device, epochs=args.epochs)
            r = evaluate_probs(te_probs, te_labs, D)
            r["seed"] = seed
            results[name].append(r)
            print(f"  seed {seed} {name:4s}: AUC={r['token_auc']:.3f}  I={r['I']:.3f}  "
                  f"drift0={r['drift0']:+.3f}  EDD={r['edd']:.1f}  "
                  f"recall={r['recall']:.2f}  @ARL0={r['arl0']:.0f}", flush=True)

    def agg(rows, key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        return float(np.nanmean(v)), float(np.nanstd(v))

    print("\n" + "=" * 70)
    print(f"H2 over {len(args.seeds)} seeds (ForwardGRU-CUSUM, ARL0={ARL0_TARGET})")
    print("=" * 70)
    summary = {}
    for name in ("base", "aug"):
        a_m, a_s = agg(results[name], "token_auc")
        I_m, I_s = agg(results[name], "I")
        g_m, g_s = agg(results[name], "drift0")
        e_m, e_s = agg(results[name], "edd")
        r_m, r_s = agg(results[name], "recall")
        summary[name] = {"token_auc": [a_m, a_s], "I": [I_m, I_s],
                         "drift0": [g_m, g_s], "edd": [e_m, e_s],
                         "recall": [r_m, r_s]}
        print(f"{name:4s}: AUC={a_m:.3f}±{a_s:.3f}  I={I_m:.3f}±{I_s:.3f}  "
              f"drift0={g_m:+.3f}±{g_s:.3f}  EDD={e_m:.1f}±{e_s:.1f}  "
              f"recall={r_m:.2f}±{r_s:.2f}")
    dA = summary["aug"]["token_auc"][0] - summary["base"]["token_auc"][0]
    dI = summary["aug"]["I"][0] - summary["base"]["I"][0]
    dG = summary["aug"]["drift0"][0] - summary["base"]["drift0"][0]
    dE = summary["aug"]["edd"][0] - summary["base"]["edd"][0]
    print(f"\nΔ aug-base:  AUC {dA:+.3f}   I {dI:+.3f}   drift0 {dG:+.3f}   EDD {dE:+.1f} tok")
    print("Mechanism (divergence up / timing down) confirmed if: AUC up-or-flat, "
          "drift0 UP (less negative), I down, EDD up.")

    json.dump({"meta": {"seeds": args.seeds, "epochs": args.epochs,
                        "D_base": D_base, "D_aug": D_aug,
                        "n_test": len(te_i)},
               "per_seed": results, "summary": summary,
               "delta": {"I": dI, "edd": dE}},
              open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
