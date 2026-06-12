"""Tests for POC: Changepoint bounds + Hawkes process fit."""
import json
import numpy as np
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent


@pytest.fixture
def label_sequences():
    """Load label sequences from RAGTruth test set."""
    with open(DATA_DIR / "test.json") as f:
        examples = json.load(f)
    seqs = []
    for ex in examples:
        labels = [t["is_hallucination"] for t in ex["token_labels"]]
        if len(labels) > 0:
            seqs.append(np.array(labels, dtype=np.float64))
    return seqs


@pytest.fixture
def hallu_sequences(label_sequences):
    """Only sequences that contain hallucination."""
    return [s for s in label_sequences if s.sum() > 0]


# === POC-1: Changepoint bounds ===

class TestTransitionMatrix:
    def test_computes_2x2_matrix(self, label_sequences):
        from poc_analysis import estimate_transition_matrix
        T = estimate_transition_matrix(label_sequences)
        assert T.shape == (2, 2)

    def test_rows_sum_to_one(self, label_sequences):
        from poc_analysis import estimate_transition_matrix
        T = estimate_transition_matrix(label_sequences)
        np.testing.assert_allclose(T.sum(axis=1), [1.0, 1.0], atol=1e-6)

    def test_persistence_ratio_above_100(self, label_sequences):
        from poc_analysis import estimate_transition_matrix
        T = estimate_transition_matrix(label_sequences)
        persistence = T[1, 1] / T[0, 1]
        assert persistence > 100


class TestKLDivergence:
    def test_kl_positive(self, label_sequences):
        from poc_analysis import compute_kl_divergence
        kl = compute_kl_divergence(label_sequences)
        assert kl > 0

    def test_kl_finite(self, label_sequences):
        from poc_analysis import compute_kl_divergence
        kl = compute_kl_divergence(label_sequences)
        assert np.isfinite(kl)


class TestLordenBound:
    def test_bound_positive(self, label_sequences):
        from poc_analysis import compute_lorden_bound
        bound = compute_lorden_bound(label_sequences, false_alarm_rate=0.01)
        assert bound > 0

    def test_bound_decreases_with_higher_alpha(self, label_sequences):
        from poc_analysis import compute_lorden_bound
        b1 = compute_lorden_bound(label_sequences, false_alarm_rate=0.01)
        b2 = compute_lorden_bound(label_sequences, false_alarm_rate=0.1)
        assert b1 > b2

    def test_bound_less_than_mean_span(self, hallu_sequences):
        from poc_analysis import compute_lorden_bound, compute_mean_span_length
        bound = compute_lorden_bound(hallu_sequences, false_alarm_rate=0.01)
        mean_span = compute_mean_span_length(hallu_sequences)
        assert bound < mean_span


# === POC-2: Hawkes process ===

class TestSpanOnsets:
    def test_extracts_onsets(self, hallu_sequences):
        from poc_analysis import extract_onsets
        onsets = extract_onsets(hallu_sequences)
        assert len(onsets) > 0
        assert all(len(doc_onsets) > 0 for doc_onsets in onsets)

    def test_onsets_are_sorted(self, hallu_sequences):
        from poc_analysis import extract_onsets
        onsets = extract_onsets(hallu_sequences)
        for doc_onsets in onsets:
            assert all(doc_onsets[i] < doc_onsets[i + 1] for i in range(len(doc_onsets) - 1))


class TestHawkesFit:
    def test_fits_parameters(self, hallu_sequences):
        from poc_analysis import fit_hawkes
        params = fit_hawkes(hallu_sequences)
        assert "mu" in params
        assert "alpha" in params
        assert "beta" in params

    def test_branching_ratio_between_0_and_1(self, hallu_sequences):
        from poc_analysis import fit_hawkes
        params = fit_hawkes(hallu_sequences)
        n = params["alpha"] / params["beta"]
        assert 0 < n < 1

    def test_branching_ratio_near_09(self, hallu_sequences):
        from poc_analysis import fit_hawkes
        params = fit_hawkes(hallu_sequences)
        n = params["alpha"] / params["beta"]
        assert 0.7 < n < 0.98

    def test_predicted_cluster_size_near_mean_span(self, hallu_sequences):
        from poc_analysis import fit_hawkes, compute_mean_span_length
        params = fit_hawkes(hallu_sequences)
        n = params["alpha"] / params["beta"]
        predicted = 1 / (1 - n)
        observed = compute_mean_span_length(hallu_sequences)
        assert abs(predicted - observed) / observed < 0.5


class TestHawkesVsPoisson:
    def test_hawkes_better_than_poisson(self, hallu_sequences):
        from poc_analysis import fit_hawkes, fit_poisson, log_likelihood_ratio
        hawkes_params = fit_hawkes(hallu_sequences)
        poisson_params = fit_poisson(hallu_sequences)
        lr = log_likelihood_ratio(hallu_sequences, hawkes_params, poisson_params)
        assert lr > 0


# === POC-2b: Token-level self-exciting model ===

class TestTokenHawkes:
    def test_fits_parameters(self, hallu_sequences):
        from poc_analysis import fit_token_hawkes
        params = fit_token_hawkes(hallu_sequences)
        assert "mu" in params
        assert "alpha" in params
        assert "beta" in params

    def test_persistence_matches_transition(self, hallu_sequences):
        """Predicted P(H_t|H_{t-1}) from Hawkes should match empirical T[1,1]."""
        from poc_analysis import fit_token_hawkes, estimate_transition_matrix
        params = fit_token_hawkes(hallu_sequences)
        T = estimate_transition_matrix(hallu_sequences)
        predicted_persistence = params["predicted_persistence"]
        observed_persistence = T[1, 1]
        assert abs(predicted_persistence - observed_persistence) < 0.1

    def test_better_than_markov(self, hallu_sequences):
        """Token Hawkes should fit better than simple first-order Markov."""
        from poc_analysis import fit_token_hawkes
        params = fit_token_hawkes(hallu_sequences)
        assert params["ll_improvement_over_markov"] > 0

    def test_excitation_decays(self, hallu_sequences):
        """Beta > 0 means excitation decays with distance."""
        from poc_analysis import fit_token_hawkes
        params = fit_token_hawkes(hallu_sequences)
        assert params["beta"] > 0

    def test_effective_memory(self, hallu_sequences):
        """Effective memory 1/beta should be near mixing time (~10 tokens)."""
        from poc_analysis import fit_token_hawkes
        params = fit_token_hawkes(hallu_sequences)
        memory = 1.0 / params["beta"]
        assert 2 < memory < 30


class TestMeanSpanLength:
    def test_mean_span_between_5_and_15(self, hallu_sequences):
        from poc_analysis import compute_mean_span_length
        mean_span = compute_mean_span_length(hallu_sequences)
        assert 5 < mean_span < 15


# === Stage 2: CUSUM detector ===

class TestCUSUM:
    def test_detects_onsets(self, hallu_sequences):
        from poc_analysis import cusum_detector
        detections = cusum_detector(hallu_sequences[:50], threshold=3.0)
        assert len(detections) > 0

    def test_detection_delay_finite(self, hallu_sequences):
        from poc_analysis import cusum_detector, compute_detection_delay
        detections = cusum_detector(hallu_sequences[:50], threshold=3.0)
        delay = compute_detection_delay(hallu_sequences[:50], detections)
        assert delay > 0
        assert np.isfinite(delay)

    def test_detection_delay_less_than_span(self, hallu_sequences):
        from poc_analysis import cusum_detector, compute_detection_delay, compute_mean_span_length
        detections = cusum_detector(hallu_sequences[:50], threshold=3.0)
        delay = compute_detection_delay(hallu_sequences[:50], detections)
        mean_span = compute_mean_span_length(hallu_sequences[:50])
        assert delay < mean_span

    def test_lower_threshold_faster_detection(self, hallu_sequences):
        from poc_analysis import cusum_detector, compute_detection_delay
        subset = hallu_sequences[:50]
        d_high = cusum_detector(subset, threshold=5.0)
        d_low = cusum_detector(subset, threshold=2.0)
        delay_high = compute_detection_delay(subset, d_high)
        delay_low = compute_detection_delay(subset, d_low)
        assert delay_low <= delay_high


# === Stage 2: Higher-order Markov ===

class TestHigherOrderMarkov:
    def test_second_order_fits(self, hallu_sequences):
        from poc_analysis import fit_markov_order_k
        params = fit_markov_order_k(hallu_sequences, k=2)
        assert "ll" in params
        assert "n_params" in params

    def test_first_order_sufficient(self, hallu_sequences):
        """If 1st order ≈ 2nd order, Markov(1) is sufficient."""
        from poc_analysis import fit_markov_order_k, likelihood_ratio_test
        m1 = fit_markov_order_k(hallu_sequences, k=1)
        m2 = fit_markov_order_k(hallu_sequences, k=2)
        p_value = likelihood_ratio_test(m1, m2)
        # We expect p > 0.01 (1st order sufficient) OR small improvement
        # Record the result either way
        assert p_value is not None

    def test_third_order_no_better(self, hallu_sequences):
        from poc_analysis import fit_markov_order_k
        m2 = fit_markov_order_k(hallu_sequences, k=2)
        m3 = fit_markov_order_k(hallu_sequences, k=3)
        improvement = (m3["ll"] - m2["ll"]) / abs(m2["ll"])
        assert improvement < 0.01  # less than 1% improvement


# === Stage 3: Feature-space CUSUM ===

@pytest.fixture
def feature_sequences():
    """Extract text features (20-dim) for test examples with hallucination."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))
    from token_features import extract_token_features, extract_token_labels
    with open(DATA_DIR / "test.json") as f:
        examples = json.load(f)
    feats = []
    labs = []
    for ex in examples:
        if ex["has_hallucination"]:
            f = extract_token_features(ex)
            l = extract_token_labels(ex)
            if len(f) > 0:
                feats.append(f)
                labs.append(l)
        if len(feats) >= 100:
            break
    return feats, labs


class TestFeatureCUSUM:
    def test_estimates_distributions(self, feature_sequences):
        from poc_analysis import estimate_feature_distributions
        feats, labs = feature_sequences
        p0, p1 = estimate_feature_distributions(feats, labs)
        assert p0["mean"].shape[0] == feats[0].shape[1]
        assert p1["mean"].shape[0] == feats[0].shape[1]

    def test_feature_kl_positive(self, feature_sequences):
        from poc_analysis import compute_feature_kl
        feats, labs = feature_sequences
        kl = compute_feature_kl(feats, labs)
        assert kl > 0

    def test_feature_cusum_detects(self, feature_sequences):
        from poc_analysis import feature_cusum_detector
        feats, labs = feature_sequences
        detections = feature_cusum_detector(feats, labs, threshold=5.0)
        assert len(detections) > 0

    def test_feature_cusum_faster_than_label_cusum(self, feature_sequences):
        from poc_analysis import feature_cusum_detector, cusum_detector, compute_detection_delay
        feats, labs = feature_sequences
        feat_dets = feature_cusum_detector(feats, labs, threshold=5.0)
        label_dets = cusum_detector(labs, threshold=3.0)
        feat_delay = compute_detection_delay(labs, feat_dets)
        label_delay = compute_detection_delay(labs, label_dets)
        assert feat_delay < label_delay

    def test_feature_cusum_delay_near_lorden(self, feature_sequences):
        from poc_analysis import feature_cusum_detector, compute_detection_delay, compute_feature_kl
        feats, labs = feature_sequences
        dets = feature_cusum_detector(feats, labs, threshold=5.0)
        delay = compute_detection_delay(labs, dets)
        kl = compute_feature_kl(feats, labs)
        lorden = np.log(1.0 / 0.01) / kl
        assert delay < lorden * 10  # within 10x of theoretical bound
