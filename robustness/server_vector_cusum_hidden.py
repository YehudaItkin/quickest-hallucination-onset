#!/usr/bin/env python3
"""Direction B2: multivariate CUSUM on the GRU HIDDEN STATE.

B1 showed multivariate accumulation on the raw 33-d features fails -- the
hallucination signal is nonlinear, so a linear/quadratic statistic on raw features
cannot reach it. B2 keeps the nonlinear extraction and adds the multivariate
accumulation where it belongs: on the learned representation h_t in R^64.

The paper's deployable detector reads ONE linear projection of h_t -- the trained
BCE head, logit = w_head' h_t -- and accumulates that scalar (the D_feat/D_inc ~ 4
bottleneck). Here we instead accumulate on the whole vector h_t:

  scalar logit-CUSUM : the paper's detector (reproduces ~11.5 tok), the baseline.
  LDA-CUSUM (h)      : the OPTIMAL linear readout of h_t, w = Sigma_h^{-1}(mu1-mu0).
                       Beats the BCE head iff cross-entropy did not find the best
                       detection projection.
  GLR-CUSUM (h)      : window-limited Hotelling GLR on h_t -- a multivariate readout
                       that uses more than one projection. Beats LDA iff a single
                       projection is not sufficient.

If LDA or GLR on h_t beats the scalar logit, the scalar bottleneck is real and
liftable by reading the representation multivariately. If not, the trained scalar
head already extracts essentially all the detection information in h_t.

Trains a ForwardGRU (BCE, seed 42), extracts hidden states, reuses the CUSUM
machinery from server_vector_cusum. Needs a GPU for training (tiny model).
"""
import json

import numpy as np
import torch

from run_extended import (
    load_and_enrich_all, assemble_features, set_seed, train_nn,
)
from server_consistency_h2 import ForwardGRU
from run_learned_cusum import cusum_reference_value, cusum_path, first_onset
from server_vector_cusum import (
    whiten_stats, lda_paths, glr_paths, evaluate, GLR_WINDOW, ARL0_MAIN, GAMMA_TARGETS,
)

VAL_FRAC = 0.15


@torch.no_grad()
def extract(model, docs, device):
    """Per-doc GRU hidden states (T, 64) and scalar logits (T,)."""
    model.eval()
    H, L = [], []
    for X in docs:
        x = torch.tensor(np.asarray(X), dtype=torch.float32, device=device).unsqueeze(0)
        h = model.gru(x)[0]                       # (1, T, 64)
        logit = model.head(h).squeeze(-1).squeeze(0)   # (T,)
        H.append(h.squeeze(0).cpu().numpy().astype(np.float64))
        L.append(logit.cpu().numpy().astype(np.float64))
    return H, L


def scalar_logit_paths(logits_list, ref):
    return [cusum_path(1.0 / (1.0 + np.exp(-lg)), ref) for lg in logits_list]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_enrich_all()
    assert data["has_all"], "need Text+NLI+LM 33-d"
    tr = assemble_features(data["tr_base"], data["tr_nli"], data["tr_lm"], True, True)
    te = assemble_features(data["te_base"], data["te_nli"], data["te_lm"], True, True)
    tr_labs, te_labs = data["tr_labs"], data["te_labs"]

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(tr))
    nval = int(VAL_FRAC * len(tr))
    val_i, tr_i = perm[:nval], perm[nval:]
    sub = lambda F, ii: [F[j] for j in ii]

    set_seed(42)
    model = ForwardGRU(tr[0].shape[1]).to(device)
    print("Training ForwardGRU (BCE, seed 42)...", flush=True)
    train_nn(model, sub(tr, tr_i), sub(tr_labs, tr_i), sub(tr, val_i), sub(tr_labs, val_i),
             te, te_labs, device, epochs=15)

    print("Extracting hidden states...", flush=True)
    Htr, _ = extract(model, tr, device)
    Hte, Lte = extract(model, te, device)
    Xtr_h = np.vstack(Htr)
    ytr = np.concatenate([np.asarray(l) for l in tr_labs])
    onsets = [first_onset(np.asarray(l)) for l in te_labs]
    n_hallu = sum(o is not None for o in onsets)
    print(f"Hidden dim={Xtr_h.shape[1]}; train {len(ytr):,} tok; test {len(Hte)} docs "
          f"({n_hallu} hallu)", flush=True)

    mu0, mu1, S = whiten_stats(Xtr_h, ytr)
    Sinv = np.linalg.inv(S)
    evals, evecs = np.linalg.eigh(S)
    Whalf = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    maha = float((mu1 - mu0) @ Sinv @ (mu1 - mu0))
    print(f"Hidden-state Mahalanobis separation = {maha:.3f} (~{maha/2:.2f} nats)\n", flush=True)

    # scalar logit CUSUM baseline (this model), reference from its test logits
    probs_te = [1.0 / (1.0 + np.exp(-lg)) for lg in Lte]
    ref, _, _ = cusum_reference_value(probs_te, te_labs)

    lda_grid = np.linspace(0.0, 300.0, 901)
    glr_grid = np.linspace(0.0, 120.0, 901)
    logit_grid = np.linspace(0.0, 120.0, 901)

    print("Detectors at matched ARL0 (censored EDD / delay-among-detected / recall):")
    out = {"hidden_maha": maha}
    out["logit-CUSUM"] = evaluate("logit-CUSUM", scalar_logit_paths(Lte, ref), onsets, logit_grid)
    out["LDA-CUSUM(h)"] = evaluate("LDA-CUSUM(h)", lda_paths(Hte, mu0, mu1, Sinv), onsets, lda_grid)
    out["GLR-CUSUM(h)"] = evaluate(f"GLR-CUSUM(h,w{GLR_WINDOW})",
                                   glr_paths(Hte, mu0, Whalf, GLR_WINDOW), onsets, glr_grid)

    print(f"\nReference (paper): ForwardGRU scalar-CUSUM 11.5 tok, Lorden floor 1.3.")
    print("Reading: LDA(h) or GLR(h) below the scalar logit-CUSUM => reading the "
          "representation multivariately lifts the scalar bottleneck.")
    json.dump(out, open("vector_cusum_hidden.json", "w"), indent=2)
    print("Saved -> vector_cusum_hidden.json")


if __name__ == "__main__":
    main()
