#!/usr/bin/env python3
"""Decompose the sequential detector's speedup into nonlinearity vs temporal
accumulation vs context, at a matched ARL0=100. Uses the tested delay_at_arl0().

  nonlinearity = LogReg-threshold      -> HistGBM-threshold
  accumulation = HistGBM-threshold     -> HistGBM-CUSUM
  context      = HistGBM-CUSUM         -> ForwardGRU-CUSUM
"""
import json
from pathlib import Path

import numpy as np

from nonlinear_baseline import delay_at_arl0

HERE = Path(__file__).resolve().parent
TARGET_ARL0 = 100


def main():
    dirp = json.load(open(HERE / "directional_probs_seed42.json"))
    hg = json.load(open(HERE / "histgbm_probs.json"))

    probs = {
        "LogReg": dirp["LogReg"]["probs"],
        "HistGBM": hg["HistGBM"]["probs"],
        "ForwardGRU": dirp["ForwardGRU"]["probs"],
    }
    labs = dirp["ForwardGRU"]["labs"]

    # alignment check: same documents, same order
    hg_labs = hg["HistGBM"]["labs"]
    assert len(hg_labs) == len(labs), "doc count mismatch"
    for a, b in zip(labs[:50], hg_labs[:50]):
        assert len(a) == len(b) and np.array_equal(np.asarray(a) > 0.5,
                                                    np.asarray(b) > 0.5), "label misalignment"
    print(f"Aligned: {len(labs)} docs\n")

    results = {}
    print(f"{'model':<12} | {'threshold delay':>16} | {'CUSUM delay':>16}")
    print("-" * 52)
    for name in ["LogReg", "HistGBM", "ForwardGRU"]:
        row = {}
        cells = []
        for mode in ["threshold", "cusum"]:
            op = delay_at_arl0(probs[name], labs, mode, TARGET_ARL0)
            row[mode] = op
            if op is None:
                cells.append("        n/a       ")
            else:
                cells.append(f"{op['delay']:5.1f} (r{op['recall']:.2f} @{op['arl0']:4.0f})")
        results[name] = row
        print(f"{name:<12} | {cells[0]:>16} | {cells[1]:>16}")

    # decomposition (delay among detected, tokens)
    def d(name, mode):
        op = results[name][mode]
        return op["delay"] if op else float("nan")

    lr_t = d("LogReg", "threshold")
    hg_t = d("HistGBM", "threshold")
    hg_c = d("HistGBM", "cusum")
    fg_c = d("ForwardGRU", "cusum")
    print("\nDecomposition of detection delay at ARL0=100 (tokens):")
    print(f"  LogReg-threshold     : {lr_t:.1f}")
    print(f"  -> HistGBM-threshold : {hg_t:.1f}   (nonlinearity: {lr_t-hg_t:+.1f})")
    print(f"  -> HistGBM-CUSUM     : {hg_c:.1f}   (accumulation: {hg_t-hg_c:+.1f})")
    print(f"  -> ForwardGRU-CUSUM  : {fg_c:.1f}   (context:      {hg_c-fg_c:+.1f})")
    print("\nReading: whichever step is largest is the real driver of the speedup.")

    json.dump({"target_arl0": TARGET_ARL0,
               "results": {n: {m: results[n][m] for m in ["threshold", "cusum"]}
                           for n in results}},
              open(HERE / "nonlinear_decomposition.json", "w"), indent=2, default=float)
    print(f"\nSaved -> nonlinear_decomposition.json")

    plot_decomposition(lr_t, hg_t, hg_c, fg_c)


def plot_decomposition(lr_t, hg_t, hg_c, fg_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = 9

    labels = ["LogReg\n(linear,\nthreshold)", "HistGBM\n(nonlinear,\nthreshold)",
              "HistGBM\n(CUSUM)", "ForwardGRU\n(CUSUM)"]
    vals = [round(v, 1) for v in (lr_t, hg_t, hg_c, fg_c)]  # round so drops are consistent
    colors = ["#7BAFD4", "#E8A33D", "#55A868", "#D64550"]
    drops = [("nonlinearity", vals[0] - vals[1]), ("accumulation", vals[1] - vals[2]),
             ("context", vals[2] - vals[3])]

    # 95% bootstrap CIs (over documents), if available
    yerr = None
    bj = HERE / "bootstrap_delay.json"
    if bj.exists():
        ci = json.load(open(bj))["cells"]
        keys = ["LogReg-threshold", "HistGBM-threshold", "HistGBM-cusum", "ForwardGRU-cusum"]
        lo = [vals[i] - ci[k][1] for i, k in enumerate(keys)]
        hi = [ci[k][2] - vals[i] for i, k in enumerate(keys)]
        yerr = np.array([lo, hi])

    _, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(4)
    ax.bar(x, vals, color=colors, width=0.6, zorder=3,
           yerr=yerr, capsize=4, error_kw=dict(ecolor="#333", lw=1))
    for xi, v in zip(x, vals):
        ax.text(float(xi) + 0.30, v, f"{v:.1f}", ha="left", va="center", fontsize=8.5)
    # annotate the drop between consecutive bars
    for i, (name, d) in enumerate(drops):
        xm = i + 0.5
        ytop = max(vals[i], vals[i + 1]) + 3.5
        ax.annotate("", xy=(i + 1, vals[i + 1] + 1.2), xytext=(i, vals[i] + 1.2),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.9))
        ax.text(xm, ytop, f"{name}\n$-${d:.1f}", ha="center", va="bottom",
                fontsize=7.5, color="gray")

    ax.axhline(NAIVE := 40.9, color="#B22222", linestyle="--", linewidth=0.9, zorder=2)
    ax.text(3.4, NAIVE - 1.5, "naive Gaussian CUSUM (40.9)", ha="right", va="top",
            fontsize=7, color="#B22222")
    ax.axhline(1.3, color="black", linestyle=":", linewidth=1.0, zorder=2)
    ax.text(3.4, 1.3 + 0.6, "Lorden floor (1.3)", ha="right", va="bottom", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Detection delay among detected (tokens)")
    ax.set_title(r"What the speedup is made of (ARL$_0$=100, $\alpha$=0.01)", fontsize=9)
    ax.set_ylim(0, 46)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    plt.tight_layout()
    out = HERE / "figures" / "delay_decomposition.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved figure -> {out}")


if __name__ == "__main__":
    main()
