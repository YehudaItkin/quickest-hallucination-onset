"""POC: Changepoint detection bounds + Hawkes process fit for hallucination."""
import numpy as np
from scipy.optimize import minimize


def estimate_transition_matrix(label_sequences):
    """Estimate 2x2 transition matrix from binary label sequences."""
    counts = np.zeros((2, 2))
    for seq in label_sequences:
        for i in range(1, len(seq)):
            prev = int(seq[i - 1])
            curr = int(seq[i])
            counts[prev, curr] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


def compute_kl_divergence(label_sequences):
    """KL divergence between post-onset and pre-onset label distributions.

    KL(P1 || P0) where P1 = distribution after hallucination onset,
    P0 = distribution before onset (stationary clean).
    Uses the transition matrix rows as the two distributions.
    """
    T = estimate_transition_matrix(label_sequences)
    p0 = T[0]  # clean -> {clean, hallu}
    p1 = T[1]  # hallu -> {clean, hallu}
    p0 = np.clip(p0, 1e-10, 1.0)
    p1 = np.clip(p1, 1e-10, 1.0)
    return float(np.sum(p1 * np.log(p1 / p0)))


def compute_lorden_bound(label_sequences, false_alarm_rate=0.01):
    """Lorden's minimax lower bound on average detection delay.

    ADD >= log(1/alpha) / KL(P1 || P0)
    """
    kl = compute_kl_divergence(label_sequences)
    if kl <= 0:
        return float("inf")
    return np.log(1.0 / false_alarm_rate) / kl


def compute_mean_span_length(label_sequences):
    """Mean length of contiguous hallucination spans."""
    lengths = []
    for seq in label_sequences:
        in_span = False
        span_len = 0
        for val in seq:
            if val > 0.5:
                if not in_span:
                    in_span = True
                    span_len = 1
                else:
                    span_len += 1
            else:
                if in_span:
                    lengths.append(span_len)
                    in_span = False
                    span_len = 0
        if in_span:
            lengths.append(span_len)
    return float(np.mean(lengths)) if lengths else 0.0


def extract_onsets(label_sequences):
    """Extract hallucination onset positions per document."""
    all_onsets = []
    for seq in label_sequences:
        onsets = []
        for i in range(len(seq)):
            if seq[i] > 0.5 and (i == 0 or seq[i - 1] < 0.5):
                onsets.append(i)
        if onsets:
            all_onsets.append(onsets)
    return all_onsets


def _hawkes_neg_log_likelihood(params, onsets_list, seq_lengths):
    """Negative log-likelihood for discrete Hawkes process on onset events."""
    mu, log_alpha, log_beta = params
    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)

    total_ll = 0.0
    for onsets, T in zip(onsets_list, seq_lengths):
        onsets = np.array(onsets, dtype=np.float64)
        for k, t_k in enumerate(onsets):
            intensity = mu
            for j in range(k):
                intensity += alpha * np.exp(-beta * (t_k - onsets[j]))
            if intensity > 0:
                total_ll += np.log(intensity)
        integral = mu * T
        for t_j in onsets:
            integral += (alpha / beta) * (1 - np.exp(-beta * (T - t_j)))
        total_ll -= integral

    return -total_ll


def fit_hawkes(label_sequences):
    """Fit Hawkes process to hallucination onsets via MLE."""
    onsets_list = extract_onsets(label_sequences)
    seq_lengths = []
    for seq in label_sequences:
        if seq.sum() > 0:
            seq_lengths.append(len(seq))

    filtered_onsets = []
    filtered_lengths = []
    idx = 0
    for seq in label_sequences:
        if seq.sum() > 0:
            onsets = []
            for i in range(len(seq)):
                if seq[i] > 0.5 and (i == 0 or seq[i - 1] < 0.5):
                    onsets.append(i)
            if onsets:
                filtered_onsets.append(onsets)
                filtered_lengths.append(len(seq))

    total_onsets = sum(len(o) for o in filtered_onsets)
    total_length = sum(filtered_lengths)
    mu_init = total_onsets / max(total_length, 1)

    x0 = [mu_init, np.log(0.5), np.log(0.2)]
    result = minimize(
        _hawkes_neg_log_likelihood,
        x0,
        args=(filtered_onsets, filtered_lengths),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )

    mu = result.x[0]
    alpha = np.exp(result.x[1])
    beta = np.exp(result.x[2])

    return {
        "mu": float(mu),
        "alpha": float(alpha),
        "beta": float(beta),
        "branching_ratio": float(alpha / beta),
        "predicted_cluster_size": float(1 / (1 - alpha / beta)) if alpha / beta < 1 else float("inf"),
        "neg_log_likelihood": float(result.fun),
        "converged": result.success,
    }


def _poisson_neg_log_likelihood(mu, onsets_list, seq_lengths):
    """Negative log-likelihood for homogeneous Poisson process."""
    total_ll = 0.0
    for onsets, T in zip(onsets_list, seq_lengths):
        n = len(onsets)
        total_ll += n * np.log(max(mu, 1e-10)) - mu * T
    return -total_ll


def fit_poisson(label_sequences):
    """Fit homogeneous Poisson process to onset events."""
    onsets_list = extract_onsets(label_sequences)
    seq_lengths = []
    for seq in label_sequences:
        if seq.sum() > 0:
            seq_lengths.append(len(seq))

    filtered_onsets = []
    filtered_lengths = []
    for seq in label_sequences:
        if seq.sum() > 0:
            onsets = []
            for i in range(len(seq)):
                if seq[i] > 0.5 and (i == 0 or seq[i - 1] < 0.5):
                    onsets.append(i)
            if onsets:
                filtered_onsets.append(onsets)
                filtered_lengths.append(len(seq))

    total_onsets = sum(len(o) for o in filtered_onsets)
    total_length = sum(filtered_lengths)
    mu_mle = total_onsets / max(total_length, 1)

    nll = _poisson_neg_log_likelihood(mu_mle, filtered_onsets, filtered_lengths)
    return {"mu": float(mu_mle), "neg_log_likelihood": float(nll)}


def log_likelihood_ratio(label_sequences, hawkes_params, poisson_params):
    """Log-likelihood ratio: Hawkes vs Poisson. Positive = Hawkes better."""
    return poisson_params["neg_log_likelihood"] - hawkes_params["neg_log_likelihood"]


# === CUSUM detector ===


def cusum_detector(label_sequences, threshold=3.0):
    """CUSUM-based hallucination onset detector.

    Uses the log-likelihood ratio between P(H_t|H_{t-1}) and P(H_t|F_{t-1})
    as the CUSUM increment. Detects when cumulative sum exceeds threshold.

    Returns list of (seq_idx, detection_position) tuples.
    """
    T = estimate_transition_matrix(label_sequences)
    log_ratio_hallu = np.log(T[1, 1] / max(T[0, 1], 1e-10))
    log_ratio_clean = np.log(T[1, 0] / max(T[0, 0], 1e-10))

    detections = []
    for seq_idx, seq in enumerate(label_sequences):
        cusum = 0.0
        detected = set()
        for t in range(len(seq)):
            if seq[t] > 0.5:
                cusum += log_ratio_hallu
            else:
                cusum += log_ratio_clean
            cusum = max(cusum, 0.0)
            if cusum >= threshold and t not in detected:
                detections.append((seq_idx, t))
                detected.add(t)
                cusum = 0.0
    return detections


def compute_detection_delay(label_sequences, detections):
    """Average detection delay: distance from true onset to CUSUM detection."""
    true_onsets = {}
    for seq_idx, seq in enumerate(label_sequences):
        onsets = []
        for i in range(len(seq)):
            if seq[i] > 0.5 and (i == 0 or seq[i - 1] < 0.5):
                onsets.append(i)
        if onsets:
            true_onsets[seq_idx] = onsets

    delays = []
    for seq_idx, det_pos in detections:
        if seq_idx in true_onsets:
            for onset in true_onsets[seq_idx]:
                if det_pos >= onset:
                    delays.append(det_pos - onset)
                    break
    return float(np.mean(delays)) if delays else float("inf")


# === Higher-order Markov ===


def fit_markov_order_k(label_sequences, k=1):
    """Fit k-th order Markov chain and return log-likelihood + param count."""
    n_states = 2
    n_contexts = n_states ** k
    counts = np.zeros((n_contexts, n_states))

    for seq in label_sequences:
        for t in range(k, len(seq)):
            context = 0
            for j in range(k):
                context = context * n_states + int(seq[t - k + j])
            curr = int(seq[t])
            counts[context, curr] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans = counts / row_sums

    total_ll = 0.0
    for seq in label_sequences:
        for t in range(k, len(seq)):
            context = 0
            for j in range(k):
                context = context * n_states + int(seq[t - k + j])
            curr = int(seq[t])
            prob = trans[context, curr]
            if prob > 0:
                total_ll += np.log(prob)

    n_params = n_contexts * (n_states - 1)
    return {"ll": float(total_ll), "n_params": n_params, "order": k}


def likelihood_ratio_test(model_simple, model_complex):
    """Likelihood ratio test. Returns p-value."""
    from scipy.stats import chi2
    lr_stat = 2 * (model_complex["ll"] - model_simple["ll"])
    df = model_complex["n_params"] - model_simple["n_params"]
    if df <= 0 or lr_stat < 0:
        return 1.0
    return float(1 - chi2.cdf(lr_stat, df))


# === Feature-space CUSUM ===


def estimate_feature_distributions(feature_seqs, label_seqs):
    """Estimate Gaussian distributions for clean (P0) and hallucinated (P1) tokens."""
    clean_feats = []
    hallu_feats = []
    for feats, labs in zip(feature_seqs, label_seqs):
        for t in range(len(labs)):
            if t < len(feats):
                if labs[t] > 0.5:
                    hallu_feats.append(feats[t])
                else:
                    clean_feats.append(feats[t])
    clean = np.array(clean_feats)
    hallu = np.array(hallu_feats)
    p0 = {"mean": clean.mean(axis=0), "std": clean.std(axis=0) + 1e-8}
    p1 = {"mean": hallu.mean(axis=0), "std": hallu.std(axis=0) + 1e-8}
    return p0, p1


def compute_feature_kl(feature_seqs, label_seqs):
    """KL divergence between hallu and clean feature distributions (diagonal Gaussian)."""
    p0, p1 = estimate_feature_distributions(feature_seqs, label_seqs)
    d = len(p0["mean"])
    kl = 0.5 * np.sum(
        np.log(p0["std"] ** 2 / p1["std"] ** 2)
        + (p1["std"] ** 2 + (p1["mean"] - p0["mean"]) ** 2) / p0["std"] ** 2
        - 1
    )
    return float(kl)


def feature_cusum_detector(feature_seqs, label_seqs, threshold=5.0):
    """CUSUM on feature-space log-likelihood ratio.

    At each token, compute log p(x|hallu) / p(x|clean) assuming diagonal Gaussian.
    Accumulate CUSUM statistic, detect when it exceeds threshold.
    """
    p0, p1 = estimate_feature_distributions(feature_seqs, label_seqs)

    detections = []
    for seq_idx, (feats, labs) in enumerate(zip(feature_seqs, label_seqs)):
        cusum = 0.0
        detected = set()
        for t in range(min(len(feats), len(labs))):
            x = feats[t]
            ll_hallu = -0.5 * np.sum(((x - p1["mean"]) / p1["std"]) ** 2 + np.log(p1["std"] ** 2))
            ll_clean = -0.5 * np.sum(((x - p0["mean"]) / p0["std"]) ** 2 + np.log(p0["std"] ** 2))
            lr = ll_hallu - ll_clean
            cusum = max(cusum + lr, 0.0)
            if cusum >= threshold and t not in detected:
                detections.append((seq_idx, t))
                detected.add(t)
                cusum = 0.0
    return detections


# === Token-level self-exciting model ===


def _token_hawkes_log_likelihood(params, label_sequences):
    """Log-likelihood for token-level self-exciting binary process.

    P(H_t=1 | history) = sigmoid(mu + sum_{s<t, H_s=1} alpha * exp(-beta * (t-s)))

    This models hallucination as a self-exciting process: each hallucinated
    token increases the probability of future tokens being hallucinated.
    """
    mu, alpha, beta = params[0], np.exp(params[1]), np.exp(params[2])

    total_ll = 0.0
    for seq in label_sequences:
        n = len(seq)
        hallu_times = []
        for t in range(n):
            excitation = 0.0
            for s in hallu_times:
                excitation += alpha * np.exp(-beta * (t - s))
            logit = mu + excitation
            prob = 1.0 / (1.0 + np.exp(-logit))
            prob = np.clip(prob, 1e-10, 1.0 - 1e-10)
            if seq[t] > 0.5:
                total_ll += np.log(prob)
                hallu_times.append(t)
            else:
                total_ll += np.log(1.0 - prob)
    return total_ll


def _markov_log_likelihood(label_sequences):
    """Log-likelihood for first-order Markov chain (baseline)."""
    T = estimate_transition_matrix(label_sequences)
    T = np.clip(T, 1e-10, 1.0)
    total_ll = 0.0
    for seq in label_sequences:
        for i in range(1, len(seq)):
            prev = int(seq[i - 1])
            curr = int(seq[i])
            total_ll += np.log(T[prev, curr])
    return total_ll


def fit_token_hawkes(label_sequences):
    """Fit token-level self-exciting model via MLE.

    For efficiency, use only sequences with hallucination and subsample if needed.
    """
    hallu_seqs = [s for s in label_sequences if s.sum() > 0]
    if len(hallu_seqs) > 200:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(hallu_seqs), 200, replace=False)
        hallu_seqs = [hallu_seqs[i] for i in indices]

    T = estimate_transition_matrix(label_sequences)
    mu_init = np.log(T[0, 1] / (1 - T[0, 1] + 1e-10))

    def neg_ll(params):
        return -_token_hawkes_log_likelihood(params, hallu_seqs)

    x0 = [mu_init, np.log(2.0), np.log(0.1)]
    result = minimize(
        neg_ll, x0, method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8},
    )

    mu = result.x[0]
    alpha = np.exp(result.x[1])
    beta = np.exp(result.x[2])

    hawkes_ll = -result.fun
    markov_ll = _markov_log_likelihood(hallu_seqs)

    sigma = lambda x: 1.0 / (1.0 + np.exp(-x))
    predicted_persistence = float(sigma(mu + alpha))

    return {
        "mu": float(mu),
        "alpha": float(alpha),
        "beta": float(beta),
        "branching_ratio": float(alpha / beta),
        "predicted_persistence": predicted_persistence,
        "effective_memory": float(1.0 / beta),
        "hawkes_ll": float(hawkes_ll),
        "markov_ll": float(markov_ll),
        "ll_improvement_over_markov": float(hawkes_ll - markov_ll),
        "converged": result.success,
    }
