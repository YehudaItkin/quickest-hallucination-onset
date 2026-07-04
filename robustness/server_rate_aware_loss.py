#!/usr/bin/env python3
"""Direction B, the real experiment: can a RATE-AWARE training objective break the
realized-rate deficit that BCE leaves on the table?

The PoC (symmetry_poc.py) showed: the CUSUM is already near-optimal given the scalar
score (D_inc/I ~ 1), and the whole 3.6x deficit is the logit being a lossy summary of
the features (D_feat/D_inc ~ 3.2). The realized rate obeys I = 2 m^2 / sigma0^2 with
m = half mean-gap of the logit and sigma0 = its clean-token std. BCE does not target
this ratio. So we add a one-sided Fisher-discriminant term that directly maximizes it:

    L = BCE(logit, y)  -  lambda * (mu1 - mu0)^2 / (Var0 + eps)

mu1 = mean logit on hallucinated tokens, mu0 = mean on clean, Var0 = clean-token logit
variance (all within the batch, masked). The term is SCALE-INVARIANT (so it does not
fight BCE on logit scale, only on shape) and it changes the learned ranking, so it is
not blocked by the paper's "invariant to monotone post-hoc reshaping" result.

We sweep lambda (0 = pure-BCE baseline = the paper's ForwardGRU), select the epoch by
held-out realized rate I, and compare I / EDD@ARL0=100 / token-AUC on the test set.
Reuses run_extended training utils and run_learned_cusum CUSUM machinery unchanged.

Run on the GPU server in ~/hallucination_exp. ForwardGRU is tiny (fits a busy GPU).
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from run_extended import (
    load_and_enrich_all, assemble_features, set_seed,
    SeqDataset, collate, score_model_full, compute_pos_weight,
)
from server_consistency_h2 import ForwardGRU
from run_learned_cusum import (
    cusum_reference_value, information_rate, cusum_path, first_onset, sweep, at_arl0,
)

VAL_FRAC = 0.15
D_FEATURE = 2.8
ARL0_TARGET = 100
CUSUM_GRID = np.linspace(0.0, 120.0, 601)     # fine grid for the final test eval
VAL_GRID = np.linspace(0.0, 120.0, 121)        # coarse grid for per-epoch val selection


def closed_rate(probs, labs):
    """Realized rate via the validated Gaussian closed form I = 2 m^2 / sigma0^2.

    Robust where the empirical Lundberg root does not exist (a rate-aware model can
    push ALL clean increments negative -> ARL0 -> inf -> brentq has no root). Returns
    I and the (m, sigma0, sigma1) that the closed form rests on.
    """
    p = np.clip(np.concatenate([np.asarray(x, float) for x in probs]), 1e-6, 1 - 1e-6)
    y = np.concatenate([np.asarray(x, float) for x in labs])
    lg = np.log(p / (1 - p))
    mu0, mu1 = lg[y < 0.5].mean(), lg[y > 0.5].mean()
    k = 0.5 * (mu0 + mu1)
    Y0, Y1 = lg[y < 0.5] - k, lg[y > 0.5] - k
    m = 0.5 * (Y1.mean() - Y0.mean())
    s0 = Y0.std() + 1e-8
    return float(2 * m * m / s0**2), float(m), float(s0), float(Y1.std())


def val_censored_edd(probs, labs):
    """Held-out censored EDD@ARL0 for model selection. Deployable and NOT gameable by
    sigma0->0 (it measures real detections), unlike the closed-form rate I."""
    ref, _, _ = cusum_reference_value(probs, labs)
    onsets = [first_onset(l) for l in labs]
    clean_idx = [i for i, o in enumerate(onsets) if o is None]
    hallu_idx = [i for i, o in enumerate(onsets) if o is not None]
    paths = [cusum_path(p, ref) for p in probs]
    rows = sweep([paths[i] for i in clean_idx], [paths[i] for i in hallu_idx],
                 [onsets[i] for i in hallu_idx], VAL_GRID)
    op = at_arl0(rows, ARL0_TARGET)
    return op["delay_edd"] if op else float("inf")


def evaluate_probs(te_probs, te_labs, D_feature):
    """Closed-form realized rate I + ARL0=100 operating point + token-AUC.

    Uses closed_rate (robust) for I; the empirical Lundberg rate is reported when it
    exists (try/except). EDD comes from the threshold sweep, which is always defined.
    """
    I_closed, m, s0, s1 = closed_rate(te_probs, te_labs)
    ref, _, _ = cusum_reference_value(te_probs, te_labs)
    try:
        I_emp = information_rate(te_probs, te_labs, ref)["I"]
    except Exception:
        I_emp = float("nan")
    onsets = [first_onset(l) for l in te_labs]
    clean_idx = [i for i, o in enumerate(onsets) if o is None]
    hallu_idx = [i for i, o in enumerate(onsets) if o is not None]
    paths = [cusum_path(p, ref) for p in te_probs]
    rows = sweep([paths[i] for i in clean_idx], [paths[i] for i in hallu_idx],
                 [onsets[i] for i in hallu_idx], CUSUM_GRID)
    op = at_arl0(rows, ARL0_TARGET)
    all_p = np.concatenate([np.asarray(p) for p in te_probs])
    all_y = np.concatenate([np.asarray(l) for l in te_labs])
    auc = float(roc_auc_score(all_y, all_p)) if len(np.unique(all_y)) > 1 else float("nan")
    return {
        "I": I_closed, "I_empirical": I_emp, "m": m, "sigma0": s0, "sigma1": s1,
        "token_auc": auc, "D_over_I": (D_feature / I_closed) if I_closed > 0 else float("inf"),
        "edd": op["delay_edd"] if op else float("nan"),
        "delay_detected": op["delay"] if op else float("nan"),
        "recall": op["recall"] if op else float("nan"),
        "arl0": op["arl0"] if op else float("nan"),
        "n_hallu_test": len(hallu_idx),
    }


def train_rate_aware(model, tr_feats, tr_labs, val_feats, val_labs, te_feats, te_labs,
                     device, lam, epochs=15):
    tr_loader = DataLoader(SeqDataset(tr_feats, tr_labs), batch_size=32, shuffle=True,
                           collate_fn=collate)
    val_loader = DataLoader(SeqDataset(val_feats, val_labs), batch_size=64,
                            shuffle=False, collate_fn=collate)
    te_loader = DataLoader(SeqDataset(te_feats, te_labs), batch_size=64,
                           shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor([compute_pos_weight(tr_labs)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    best_edd, best_state = float("inf"), None

    for _ in range(epochs):
        model.train()
        for feats, labels, _ in tr_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            mask = labels >= 0
            bce_loss = (bce(logits, labels.clamp(min=0)) * mask).sum() / mask.sum()
            # Penalize the CLEAN-token logit variance (sigma0) directly. BCE fixes
            # direction and the mean-gap m; this term quiets the pre-change baseline,
            # which raises the realized rate I = 2 m^2 / sigma0^2. Unlike the discriminant
            # ratio, a variance penalty cannot invert the score or game a denominator->0.
            clean = mask & (labels < 0.5)
            var0 = logits[clean].var() if clean.sum() > 1 else torch.zeros((), device=device)
            loss = bce_loss + lam * var0
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # model selection by held-out censored EDD (deployable; not gameable)
        vp, vl = score_model_full(model, val_loader, device)
        edd_val = val_censored_edd(vp, vl)
        if edd_val < best_edd:
            best_edd = edd_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return score_model_full(model, te_loader, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="*", default=[0.0, 0.3, 1.0, 3.0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="rate_aware_loss.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    data = load_and_enrich_all()
    assert data["has_all"], "need Text+NLI+LM for the 33-d base"
    tr = assemble_features(data["tr_base"], data["tr_nli"], data["tr_lm"], True, True)
    te = assemble_features(data["te_base"], data["te_nli"], data["te_lm"], True, True)
    tr_labs, te_labs = data["tr_labs"], data["te_labs"]

    # fixed val carve-out from train (same across lambda/seed for fair comparison)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(tr))
    nval = int(VAL_FRAC * len(tr))
    val_i, tr_i = perm[:nval], perm[nval:]
    sub = lambda F, ii: [F[j] for j in ii]
    print(f"Train {len(tr_i)} / val {len(val_i)} / test {len(te)} docs; dim={tr[0].shape[1]}",
          flush=True)

    # resume: load any completed (lambda, seed) runs so a kill (shared GPU) is cheap
    from pathlib import Path
    results = {str(l): [] for l in args.lambdas}
    if Path(args.out).exists():
        try:
            prev = json.load(open(args.out)).get("per_run", {})
            for k, rows in prev.items():
                if k in results:
                    results[k] = rows
            done = sum(len(v) for v in results.values())
            print(f"Resuming: {done} runs already saved in {args.out}", flush=True)
        except Exception:
            pass

    def save():
        json.dump({"meta": {"lambdas": args.lambdas, "seeds": args.seeds,
                            "epochs": args.epochs, "D_feature": D_FEATURE},
                   "per_run": results}, open(args.out, "w"), indent=2)

    for lam in args.lambdas:
        have = {r["seed"] for r in results[str(lam)]}
        for seed in args.seeds:
            if seed in have:
                continue
            set_seed(seed)
            model = ForwardGRU(tr[0].shape[1]).to(device)
            tp, tl = train_rate_aware(
                model, sub(tr, tr_i), sub(tr_labs, tr_i),
                sub(tr, val_i), sub(tr_labs, val_i), te, te_labs,
                device, lam, args.epochs)
            r = evaluate_probs(tp, tl, D_FEATURE)
            r["seed"] = seed
            results[str(lam)].append(r)
            save()   # incremental: never lose more than the current run
            print(f"  lam={lam:<4} seed {seed}: AUC={r['token_auc']:.3f}  I={r['I']:.3f}  "
                  f"D/I={r['D_over_I']:.1f}x  EDD={r['edd']:.1f}  recall={r['recall']:.2f}  "
                  f"@ARL0={r['arl0']:.0f}", flush=True)

    def agg(rows, k):
        v = np.array([x[k] for x in rows], float)
        return float(np.nanmean(v)), float(np.nanstd(v))

    print("\n" + "=" * 74)
    print(f"Rate-aware loss sweep (ForwardGRU-CUSUM, ARL0=100, {len(args.seeds)} seeds)")
    print("=" * 74)
    summary = {}
    for lam in args.lambdas:
        rows = results[str(lam)]
        a = agg(rows, "token_auc"); I = agg(rows, "I")
        e = agg(rows, "edd"); rc = agg(rows, "recall"); mm = agg(rows, "m"); s0 = agg(rows, "sigma0")
        summary[str(lam)] = {"token_auc": a, "I": I, "edd": e, "recall": rc,
                             "m": mm, "sigma0": s0}
        tag = "(BCE baseline)" if lam == 0.0 else ""
        print(f"lam={lam:<4}: AUC={a[0]:.3f}±{a[1]:.3f}  I={I[0]:.3f}±{I[1]:.3f}  "
              f"m={mm[0]:.3f} s0={s0[0]:.3f}  EDD={e[0]:.1f}±{e[1]:.1f}  recall={rc[0]:.2f} {tag}")
    print("\nGoal: a lambda>0 with I above the BCE baseline beyond seed spread, and lower EDD.")

    json.dump({"meta": {"lambdas": args.lambdas, "seeds": args.seeds,
                        "epochs": args.epochs, "D_feature": D_FEATURE},
               "per_run": results, "summary": summary},
              open(args.out, "w"), indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
