# Quickest Detection of Hallucination Onset

Code for the paper **"Quickest Detection of Hallucination Onset: Delay Bounds and
Learned CUSUM Statistics."**

> 📄 Paper: [arXiv:2606.12476](https://arxiv.org/abs/2606.12476)
> 👤 Igor Itkin ([ORCID 0009-0004-9513-8463](https://orcid.org/0009-0004-9513-8463))

This repository contains the analysis that frames token-level hallucination
detection as **sequential change-point detection**. The hallucination onset is the
change point; tokens are the stream. We compare a learned causal statistic (a
ForwardGRU read as a *learned CUSUM*) against the Lorden minimax delay bound at a
matched false-alarm rate (ARL₀), and decompose the remaining gap to the floor.

Everything here is **CPU-only and runs in seconds** — there is no training and no
GPU in this repository. The scripts operate on saved per-token posteriors; see
[Reproducing the inputs](#reproducing-the-inputs) for how those are produced.

## What's here

| File | What it computes |
|---|---|
| `run_learned_cusum.py` | **Headline.** ARL₀-matched sequential detection; the Lorden bound as a curve `EDD_min(γ)=ln(γ)/KL`; ForwardGRU-threshold and explicit ForwardGRU-CUSUM operating points; the realized information rate `I(ĝ)=ω·E₁[Y]` (Lundberg coefficient ω) and the `D/I` gap to the floor. Writes `figures/learned_cusum_operating_point.pdf`. |
| `run_nonlinear_decomposition.py` | Decomposes the detector's speedup into **nonlinearity → accumulation → temporal context** at matched ARL₀=100, using a nonlinear per-token model (HistGBM) as the intermediate control. Writes `figures/delay_decomposition.pdf`. |
| `bootstrap_delay.py` | Conditional bootstrap 95% CIs for the four operating points and the three paired drops of the decomposition. |
| `gap_analysis.py` | What does (and does not) close the gap to the Lorden floor: temperature-scaling invariance, isotonic recalibration, and the integrated autocorrelation of the score. |
| `info_theory_limits.py` | Captured mutual information `I(p̂;Y)=H(Y)−H(Y|p̂)` (calibration-free) for LogReg / ForwardGRU / BiGRU — the information reading of the architecture-independent ceiling. |
| `pac_bayes_poc.py` | Data-side sample-complexity quantities: temporal redundancy `1−h/H(Y)`, mixing time, integrated autocorrelation τ, and the effective sample size `N_eff = N/τ`. |
| `poc_analysis.py` | Library: transition-matrix / KL / Lorden-bound estimators, Hawkes and higher-order Markov fits, and a feature-space CUSUM. |

Tests: `test_run_learned_cusum.py`, `test_nonlinear_baseline.py` (self-contained,
synthetic, expected values computed by hand); `test_poc.py`, `test_data_integrity.py`
(require the data described below).

## Installation

```bash
pip install -r requirements.txt
```

## Reproducing the inputs

The analysis scripts read **saved per-token posteriors** from this directory:

- `directional_probs_seed42.json` — per-token P(hallucination) for `LogReg`,
  `ForwardGRU`, `BiGRU` (and `BackwardGRU`) on the RAGTruth test split, plus the
  gold token labels. Required by every analysis script.
- `histgbm_probs.json` — per-token posteriors from a nonlinear per-token model
  (HistGradientBoosting), in the same document order. Required by
  `run_nonlinear_decomposition.py` and `bootstrap_delay.py`.

These files are **not redistributed** here (they are derived from RAGTruth, which
carries its own license, and all `*.json` is git-ignored). To regenerate them:

1. Obtain the **RAGTruth** token-labeled dataset
   (https://github.com/ParticleMedia/RAGTruth).
2. Build the 33-dim feature stream (text + NLI + LM signals) and train the
   per-token / recurrent models with the companion repository
   **`temporal-hallucination-detection`** (released with the companion paper)
   (`run_extended.py` for the feature pipeline and training; `ForwardGRU` /
   `BiGRU` are in `src/models.py`). Dump the per-token test posteriors to
   `directional_probs_seed42.json` (and the HistGBM posteriors to
   `histgbm_probs.json`) and place them in this directory.
3. The data-integrity test (`test_data_integrity.py`) verifies that the saved
   labels match RAGTruth and that all posteriors are aligned to the same
   documents, so you can confirm the inputs before trusting the numbers.

> The upstream feature extraction, model training, and the multi-signal fusion
> detector live in the companion repository above; this repository is the
> sequential-detection / delay-bound analysis built on its outputs.

## Running

```bash
python run_learned_cusum.py            # headline operating-point table + figure
python run_nonlinear_decomposition.py  # speedup decomposition (needs histgbm_probs.json)
python bootstrap_delay.py              # bootstrap CIs for the decomposition
python gap_analysis.py                 # what closes the gap to the Lorden floor
python info_theory_limits.py           # captured-MI ceiling
python pac_bayes_poc.py                # effective sample size / redundancy
```

## Robustness analysis (paper Section 4.4 and Appendix F)

The [`robustness/`](robustness/) directory holds the server-side experiment scripts
behind the v3 robustness section: the three levers that could break the 4.5x deficit
(richer self-consistency features, a rate-aware objective, a multivariate accumulator)
and why each fails, plus the covariance-term detector and additional probes. See
[`robustness/README.md`](robustness/README.md) for the script-to-claim map.

## Tests

```bash
pytest test_run_learned_cusum.py test_nonlinear_baseline.py   # no data required
pytest test_poc.py test_data_integrity.py                     # require the inputs above
```

## Citation

```bibtex
@misc{itkin2026quickest,
  title  = {Quickest Detection of Hallucination Onset: Delay Bounds and Learned CUSUM Statistics},
  author = {Itkin, Igor},
  year   = {2026},
  eprint = {2606.12476},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```

## License

MIT — see [LICENSE](LICENSE).
