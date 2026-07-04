# Robustness of the deficit (paper Section 4.4 and Appendix F)

Analysis code behind the v3 robustness section: the three natural levers that
*could* break the 4.5x realized-rate deficit — richer black-box features, a
rate-aware training objective, and a multivariate accumulator — and why each fails.

These are the server-side experiment scripts. Like the core release, they are
illustrative: they run against the RAGTruth feature pipeline and saved detector
posteriors of the companion multi-signal system, not standalone.

## Rate anatomy and rate-aware objective (Sec. 4.4, App. F Table 3/4)
- `symmetry_poc.py` — validates the closed form `I = 2 m^2 / sigma0^2` on the saved
  increments, and the deficit factorization `D/I = (D/D_inc)(D_inc/I)`.
- `server_rate_aware_loss.py` — the variance-penalty (rate-aware) training sweep;
  `m/sigma0 ~ 0.57` stays scale-invariant, censored EDD does not move (null).

## Feature augmentation: self-consistency (Sec. 4.4, App. F)
- `server_consistency_features.py` — generates K=5 resamples + SelfCheckGPT-NLI
  per-token consistency features.
- `server_consistency_divergence.py` — kNN-KL conditional divergence the block adds
  (+0.83 nats), vs a random-feature control.
- `server_consistency_h2.py` — trains ForwardGRU-CUSUM on base vs base+consistency
  (five seeds); the null on AUC / I / censored EDD.
- `server_consistency_redundancy.py` — how much of the consistency signal the base
  score already predicts (R^2).
- `test_consistency_alignment.py` — CPU unit tests for the consistency-feature/token
  alignment.

## Multivariate accumulation (Sec. 4.4, App. F vector table)
- `server_vector_cusum.py` — feature-space LDA-CUSUM and GLR-CUSUM.
- `server_vector_cusum_hidden.py` — the same on the recurrent hidden state (h in R^64);
  LDA readout matches the scalar head, Hotelling GLR is worse.

## Covariance-term detector (Sec. 4.4 footnote)
- `server_covariance_ablate.py` — quad-only vs linear-only vs const-drift ablation.
- `server_covariance_final.py` — recall and delay-among-detected vs the best scalar,
  reported separately.
- `server_covariance_controls.py` — leave-one-generator-out + paired bootstrap
  (+0.065 recall, 95% CI [0.03, 0.10], open-weight generators).
- `server_covariance_validate.py` — stability + OOD checks.

## Additional detectors probing the deficit (background)
- `server_shiryaev.py` — Shiryaev-Roberts / Bayesian Shiryaev filter (known hazard
  does not help).
- `server_recall_schemes.py` — recall vs ARL0, persistence-k debounce.
- `server_variance_cusum.py` — energy / dispersion CUSUM.
- `server_combined_detector.py` — scalar + covariance combination (does not help).

## Shared divergence / decomposition helpers (also back v2)
- `server_knn_divergence.py` — kNN-KL feature divergence estimate.
- `server_seed_decomposition.py` — seed-averaged speedup decomposition.

Paper: arXiv:2606.12476.
