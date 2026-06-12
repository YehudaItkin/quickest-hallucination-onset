#!/usr/bin/env python3
"""
Learned CUSUM: how close does the recurrent causal statistic get to the
theoretical Lorden bound on detection delay?

We frame hallucination onset detection as sequential change-point detection
(Lorden/Pollak single change per document):
  - changepoint theta = first hallucination onset in the document
  - an alarm at t < theta is a FALSE ALARM (stopped before the change)
  - an alarm at t >= theta is a DETECTION with delay (t - theta)

All detectors are compared at a MATCHED operating point measured in the SAME
units as the Lorden bound: the average run length to false alarm,

  ARL0 = mean number of clean tokens until the first (false) alarm,

estimated on the fully-clean documents (censored at document end). Lorden's
minimax bound is then a CURVE, not a point:

  EDD_min(gamma) = ln(gamma) / KL(P1 || P0),   gamma = ARL0.

The canonical alpha=0.01 operating point of the POC corresponds to ARL0=100
(EDD_min ~ 1.3 tokens). Using ARL0 (a per-step rate) rather than a per-document
false-alarm rate is the methodological fix: a per-document rate is ~L times
stricter (L = document length), which is why naive per-document thresholding
looked hopeless. The expected detection delay (EDD) is censored over ALL
hallucination documents, so it cannot be inflated by low recall.

Detectors (all CAUSAL, see only the past -> valid for online detection):
  - oracle-label-CUSUM : LLR increment from the TRUE-label transition matrix.
                         Reference: best a label-space detector can do.
  - LogReg-threshold   : per-token posterior, no temporal accumulation (ablation).
  - ForwardGRU-threshold : recurrent state already accumulates evidence; we just
                         threshold the per-token posterior. This IS a learned CUSUM.
  - ForwardGRU-CUSUM   : explicit CUSUM on the model log-odds increment
                         logit(p_t) - logit(pi), pi = prior hallu rate.

Reference points (from POC, not on this curve):
  - Lorden bound (alpha=0.01)            = 1.31 tok
  - naive Gaussian feature CUSUM (33-d)  = 40.9 tok
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 9

# Self-contained: reads the saved causal per-token posteriors from this directory
# (see README for how to regenerate directional_probs_seed42.json) and writes its
# results and figure here. No GPU required.
HERE = Path(__file__).resolve().parent
PROBS_FILE = HERE / "directional_probs_seed42.json"
FIGURES = HERE / "figures"
OUT_RESULTS = HERE / "learned_cusum.json"

LORDEN_BOUND = 1.31          # tokens, alpha=0.01 (ARL0=100), 33-dim feature KL (POC)
NAIVE_GAUSSIAN_CUSUM = 40.9  # tokens, POC stage 3
ARL0_TARGETS = [50, 100, 200]  # 100 == Lorden's alpha=0.01 operating point


# ----------------------------------------------------------------------------
# Per-document change-point bookkeeping
# ----------------------------------------------------------------------------

def first_onset(labs):
    """Index of the first 0->1 transition (changepoint theta). None if clean."""
    for t in range(len(labs)):
        if labs[t] > 0.5:
            return t
    return None


def first_crossing(stat_path, threshold):
    """First index where the (already causal) statistic crosses threshold."""
    for t, s in enumerate(stat_path):
        if s >= threshold:
            return t
    return None


# ----------------------------------------------------------------------------
# Causal statistic paths (one per document)
# ----------------------------------------------------------------------------

def prob_path(probs):
    """The posterior itself is the statistic (recurrence already accumulated)."""
    return np.asarray(probs, dtype=np.float64)


def cusum_path(probs, ref):
    """Explicit CUSUM on the model log-odds increment, reset at 0.

    increment_t = logit(p_t) - ref,  S_t = max(0, S_{t-1} + increment_t)
    ref is the textbook CUSUM reference value k = (mu0 + mu1) / 2, the midpoint
    between the mean clean (mu0) and mean hallucinated (mu1) log-odds. This makes
    a clean token give a negative increment on average and a hallucinated token a
    positive one, regardless of the model's (mis)calibration.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    S = 0.0
    out = np.empty(len(p))
    for t in range(len(p)):
        S = max(0.0, S + (logit[t] - ref))
        out[t] = S
    return out


def cusum_reference_value(probs_list, labs_list):
    """Textbook CUSUM reference k = (mu0 + mu1)/2 in the log-odds domain."""
    clean, hallu = [], []
    for probs, labs in zip(probs_list, labs_list):
        p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
        lg = np.log(p / (1 - p))
        lab = np.asarray(labs)
        clean.append(lg[lab < 0.5])
        hallu.append(lg[lab > 0.5])
    mu0 = float(np.concatenate(clean).mean())
    mu1 = float(np.concatenate(hallu).mean())
    return 0.5 * (mu0 + mu1), mu0, mu1


def information_rate(probs_list, labs_list, ref):
    """Realized information rate I(g) = omega * E_1[Y] of the learned score.

    For a CUSUM on i.i.d. increments Y = logit(p_hat) - ref with pre-change drift
    E_0[Y] < 0 < E_1[Y], the first-order delay is EDD ~ ln(ARL0)/I where
    I = omega * E_1[Y] and omega > 0 is the adjustment (Lundberg) coefficient
    solving E_0[exp(omega Y)] = 1 (Siegmund 1985). For the true log-likelihood
    ratio omega = 1 and I = D(P1||P0); for any other score I <= D (LLR-CUSUM is
    delay-optimal, Moustakides 1986). The gap to the floor is D / I.
    """
    from scipy.optimize import brentq
    clean, hallu = [], []
    for probs, labs in zip(probs_list, labs_list):
        p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
        lg = np.log(p / (1 - p))
        lab = np.asarray(labs)
        clean.append(lg[lab < 0.5] - ref)   # pre-change increments
        hallu.append(lg[lab > 0.5] - ref)   # post-change increments
    Yc = np.concatenate(clean)
    Yh = np.concatenate(hallu)
    delta1 = float(Yh.mean())

    def mgf(w):
        return float(np.mean(np.exp(w * Yc)) - 1.0)

    hi = 0.5
    while mgf(hi) < 0 and hi < 50:
        hi *= 1.5
    omega = float(brentq(mgf, 1e-6, hi))  # type: ignore[arg-type]
    return {"omega": omega, "delta1": delta1, "I": omega * delta1,
            "drift0": float(Yc.mean())}


def label_cusum_path(labs, T):
    """Oracle label-space CUSUM using the true-label transition matrix LLR."""
    lr_hallu = np.log(max(T[1, 1], 1e-10) / max(T[0, 1], 1e-10))
    lr_clean = np.log(max(T[1, 0], 1e-10) / max(T[0, 0], 1e-10))
    S = 0.0
    out = np.empty(len(labs))
    for t in range(len(labs)):
        inc = lr_hallu if labs[t] > 0.5 else lr_clean
        S = max(0.0, S + inc)
        out[t] = S
    return out


def estimate_transition_matrix(label_seqs):
    counts = np.zeros((2, 2))
    for seq in label_seqs:
        for i in range(1, len(seq)):
            counts[int(seq[i - 1]), int(seq[i])] += 1
    rs = counts.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return counts / rs


# ----------------------------------------------------------------------------
# Evaluate a set of precomputed statistic paths at one threshold
# ----------------------------------------------------------------------------

def arl0_on_clean_stream(clean_paths, threshold):
    """ARL0 = total clean tokens / number of alarms on the concatenated clean
    stream (reset after each alarm). For a memoryless threshold statistic this
    equals 1/p_fa; for a CUSUM path it counts threshold up-crossings. Not capped
    by document length (unlike a per-document run length).
    """
    total = 0
    alarms = 0
    for path in clean_paths:
        above = path >= threshold
        total += len(path)
        # count up-crossings (transitions False->True), i.e. distinct alarms
        if len(above):
            alarms += int(above[0]) + int(np.sum(above[1:] & ~above[:-1]))
    return total / alarms if alarms > 0 else float("inf")


def edd_on_hallu(hallu_paths, hallu_onsets, threshold):
    """Detection statistics over hallucination documents at a given threshold.

    delay_edd : censored EDD over ALL hallu docs (miss/pre-change FA -> len-theta).
    delay     : mean over DETECTED docs only (reference).
    recall    : fraction detected at or after theta.
    """
    edd, delays = [], []
    n_detected = 0
    for path, theta in zip(hallu_paths, hallu_onsets):
        stop = first_crossing(path, threshold)
        max_delay = len(path) - theta
        if stop is None or stop < theta:
            edd.append(max_delay)
        else:
            n_detected += 1
            delays.append(stop - theta)
            edd.append(stop - theta)
    mean_edd = float(np.mean(edd)) if edd else float("nan")
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    recall = n_detected / max(len(hallu_paths), 1)
    return mean_edd, mean_delay, recall


def sweep(clean_paths, hallu_paths, hallu_onsets, thresholds):
    rows = []
    for thr in thresholds:
        arl0 = arl0_on_clean_stream(clean_paths, thr)
        edd, delay, recall = edd_on_hallu(hallu_paths, hallu_onsets, thr)
        rows.append({"threshold": float(thr), "arl0": arl0,
                     "delay_edd": edd, "delay": delay, "recall": recall})
    return rows


def at_arl0(rows, arl0_target):
    """Pick the operating point whose ARL0 is closest to arl0_target.

    Among rows with arl0 >= target (at least the required false-alarm budget),
    choose the smallest arl0 (most sensitive). Falls back to nearest ARL0.
    """
    valid = [r for r in rows if not np.isnan(r["delay_edd"]) and not np.isnan(r["arl0"])]
    if not valid:
        return None
    above = [r for r in valid if r["arl0"] >= arl0_target]
    if above:
        return min(above, key=lambda r: r["arl0"])
    return min(valid, key=lambda r: abs(r["arl0"] - arl0_target))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    data = json.load(open(PROBS_FILE))
    fwd = data["ForwardGRU"]
    logreg = data["LogReg"]

    labs_list = [np.asarray(l, dtype=np.float64) for l in fwd["labs"]]
    onsets = [first_onset(l) for l in labs_list]
    n_hallu = sum(o is not None for o in onsets)
    pi = float(np.concatenate(labs_list).mean())
    ref, mu0, mu1 = cusum_reference_value(fwd["probs"], fwd["labs"])
    print(f"Docs: {len(labs_list)}, hallu docs: {n_hallu}, "
          f"hallu-token rate pi={pi:.4f}")
    print(f"ForwardGRU log-odds: clean mu0={mu0:.3f}, hallu mu1={mu1:.3f}, "
          f"CUSUM reference k=(mu0+mu1)/2={ref:.3f}")

    # Realized information rate and the two-factor gap decomposition
    irate = information_rate(fwd["probs"], fwd["labs"], ref)
    D_FEATURE = 3.5  # diagonal-Gaussian feature KL (nats), POC stage 3
    lorden100 = np.log(100) / D_FEATURE
    pred_edd100 = np.log(100) / irate["I"]
    print(f"Realized info rate: omega={irate['omega']:.3f}, delta1={irate['delta1']:.3f}, "
          f"I(g_hat)={irate['I']:.3f} nats/token")
    print(f"  Lorden floor (D={D_FEATURE} nats): EDD@100 = {lorden100:.2f} tok")
    print(f"  predicted EDD@100 from I(g_hat) = {pred_edd100:.2f} tok  "
          f"(realized-divergence factor D/I = {D_FEATURE/irate['I']:.1f}x)")

    # --- build causal statistic paths per detector ---
    fwd_probs = [prob_path(p) for p in fwd["probs"]]
    lr_probs = [prob_path(p) for p in logreg["probs"]]
    fwd_cusum = [cusum_path(p, ref) for p in fwd["probs"]]

    T = estimate_transition_matrix(labs_list)
    label_cusum = [label_cusum_path(l, T) for l in labs_list]
    p0 = np.clip(T[0], 1e-10, 1.0)
    p1 = np.clip(T[1], 1e-10, 1.0)
    kl_label = float(np.sum(p1 * np.log(p1 / p0)))   # nats, for Lorden curve
    print(f"Transition matrix: P(F->H)={T[0,1]:.4f}, P(H->H)={T[1,1]:.4f}")
    print(f"Label KL(P1||P0)={kl_label:.3f} nats -> Lorden EDD(ARL0=100)="
          f"{np.log(100)/kl_label:.2f} tok")

    # --- threshold grids ---
    prob_grid = np.unique(np.concatenate([
        np.linspace(0.0, 0.999, 400),
        1 - np.logspace(-5, -1, 200),   # dense near 1.0 for large ARL0
    ]))
    cusum_grid = np.linspace(0.0, 120.0, 601)
    label_grid = np.linspace(0.0, 60.0, 401)

    detectors = {
        "oracle-label-CUSUM": (label_cusum, label_grid),
        "LogReg-threshold": (lr_probs, prob_grid),
        "ForwardGRU-threshold": (fwd_probs, prob_grid),
        "ForwardGRU-CUSUM": (fwd_cusum, cusum_grid),
    }

    clean_idx = [i for i, o in enumerate(onsets) if o is None]
    hallu_idx = [i for i, o in enumerate(onsets) if o is not None]
    hallu_onsets = [onsets[i] for i in hallu_idx]
    print(f"clean docs: {len(clean_idx)}, hallu docs: {len(hallu_idx)}")

    curves = {}
    for name, (paths, grid) in detectors.items():
        clean_paths = [paths[i] for i in clean_idx]
        hallu_paths = [paths[i] for i in hallu_idx]
        curves[name] = sweep(clean_paths, hallu_paths, hallu_onsets, grid)
        print(f"  swept {name}: {len(grid)} thresholds")

    # --- operating points at matched ARL0 (Lorden's gamma) ---
    print("\n" + "=" * 96)
    print("Operating points at matched ARL0 (mean clean tokens to false alarm)")
    print("cells = censored-EDD / delay-among-detected / recall  @actual-ARL0")
    print("=" * 96)
    print(f"{'Detector':<22} | " + " | ".join(f"ARL0~{g}" for g in ARL0_TARGETS))
    print("-" * 96)
    table = {}
    for name, rows in curves.items():
        cells = []
        table[name] = {}
        for g in ARL0_TARGETS:
            op = at_arl0(rows, g)
            if op is None:
                cells.append("        n/a         ")
                table[name][g] = None
            else:
                cells.append(f"{op['delay_edd']:5.1f} / {op['delay']:5.1f} / "
                             f"{op['recall']:.2f} @{op['arl0']:5.0f}")
                table[name][g] = op
        if name == "oracle-label-CUSUM":
            continue  # degenerate reference, printed separately below
        print(f"{name:<22} | " + " | ".join(cells))
    print("-" * 96)
    print(f"{'oracle-label-CUSUM':<22} | observes true labels: EDD=0, recall=1.0 "
          f"at ARL0=inf (detects every onset immediately)")
    for g in ARL0_TARGETS:
        print(f"  Lorden EDD_min(ARL0={g:<4d}) = {np.log(g)/kl_label:.2f} tok")
    print(f"  POC reference: naive Gaussian feature CUSUM = {NAIVE_GAUSSIAN_CUSUM:.1f} tok")

    # --- save ---
    OUT_RESULTS.parent.mkdir(exist_ok=True)
    json.dump({
        "meta": {
            "n_docs": len(labs_list), "n_hallu_docs": n_hallu,
            "pi": pi, "cusum_ref": ref, "mu0_clean": mu0, "mu1_hallu": mu1,
            "transition_matrix": T.tolist(), "kl_label_nats": kl_label,
            "lorden_bound_arl0_100": float(np.log(100) / kl_label),
            "naive_gaussian_cusum": NAIVE_GAUSSIAN_CUSUM,
            "arl0_targets": ARL0_TARGETS,
            "info_rate": irate, "D_feature_nats": D_FEATURE,
            "predicted_edd_100": pred_edd100,
        },
        "operating_points": {
            name: {str(g): table[name][g] for g in ARL0_TARGETS}
            for name in detectors
        },
        "curves": curves,
    }, open(OUT_RESULTS, "w"), indent=2)
    print(f"\nSaved -> {OUT_RESULTS}")

    # --- figure: clean bar chart at the ARL0=100 operating point ---
    plot_operating_point(table, kl_label, arl0=100)


def plot_operating_point(table, kl_label, arl0=100):
    """Bar chart of detection delay at the ARL0 operating point (= Lorden's gamma).

    Bars = delay among detected onsets (the speed when the detector fires).
    Recall is annotated above each bar. Horizontal lines mark the Lorden bound
    and the POC naive Gaussian feature CUSUM.
    """
    order = ["LogReg-threshold", "ForwardGRU-threshold", "ForwardGRU-CUSUM"]
    labels = ["LogReg\n(per-token)", "ForwardGRU\n(threshold)", "ForwardGRU\n(CUSUM)"]
    colors = ["#7BAFD4", "#D64550", "#55A868"]

    delays = [table[n][arl0]["delay"] for n in order]
    recalls = [table[n][arl0]["recall"] for n in order]
    lorden = float(np.log(arl0) / kl_label)

    _, ax = plt.subplots(figsize=(4.8, 3.4))
    x = np.arange(len(order))
    ax.bar(x, delays, color=colors, width=0.62, zorder=3)
    for xi, d, r in zip(x, delays, recalls):
        ax.text(float(xi), d + 0.8, f"{d:.1f} tok\n(recall {r:.0%})",
                ha="center", va="bottom", fontsize=7.5)

    ax.axhline(NAIVE_GAUSSIAN_CUSUM, color="#B22222", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(len(order) - 0.5, NAIVE_GAUSSIAN_CUSUM - 1.6,
            f"naive Gaussian CUSUM ({NAIVE_GAUSSIAN_CUSUM:.0f})",
            ha="right", va="top", fontsize=7, color="#B22222")
    ax.axhline(lorden, color="black", linestyle=":", linewidth=1.1, zorder=2)
    ax.text(len(order) - 0.5, lorden + 0.6, f"Lorden bound ({lorden:.1f})",
            ha="right", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Detection delay among detected onsets (tokens)")
    ax.set_title(rf"Sequential onset detection at ARL$_0$={arl0} ($\alpha$=0.01)",
                 fontsize=9)
    ax.set_ylim(0, NAIVE_GAUSSIAN_CUSUM + 6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "learned_cusum_operating_point.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved figure -> {out}")


if __name__ == "__main__":
    main()
