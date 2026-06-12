"""Characterization tests for the run_learned_cusum detectors.

These functions carry BOTH the sequential paper's headline numbers (LogReg 30.8,
ForwardGRU 13.4/11.5) and the nonlinear decomposition. Expected values are
computed BY HAND, not read from the implementation, so a test failure means the
logic is wrong (not merely changed).
"""
import math

import numpy as np
import pytest

import run_learned_cusum as L

SIG1 = 1.0 / (1.0 + math.exp(-1.0))   # = 0.7310585..., logit(SIG1) == 1.0 exactly


# ---- first_onset ----------------------------------------------------------
def test_first_onset_basic():
    assert L.first_onset(np.array([0, 0, 1, 1, 0])) == 2
    assert L.first_onset(np.array([1, 0, 0])) == 0
    assert L.first_onset(np.array([0, 0, 0])) is None


# ---- first_crossing -------------------------------------------------------
def test_first_crossing_inclusive_and_none():
    assert L.first_crossing(np.array([0.1, 0.3, 0.6, 0.2]), 0.5) == 2
    assert L.first_crossing(np.array([0.1, 0.3, 0.4]), 0.5) is None
    assert L.first_crossing(np.array([0.5]), 0.5) == 0        # >= is inclusive


# ---- cusum_path -----------------------------------------------------------
def test_cusum_path_accumulates_and_resets():
    # logit(SIG1)=+1, logit(1-SIG1)=-1; ref=0
    # probs=[1-SIG1, SIG1, SIG1]: increments -1,+1,+1
    # S: max(0,-1)=0, max(0,0+1)=1, max(0,1+1)=2
    out = L.cusum_path([1 - SIG1, SIG1, SIG1], ref=0.0)
    assert np.allclose(out, [0.0, 1.0, 2.0], atol=1e-6)

def test_cusum_path_floor_at_zero():
    out = L.cusum_path([1 - SIG1, 1 - SIG1], ref=0.0)   # increments -1,-1
    assert np.allclose(out, [0.0, 0.0], atol=1e-6)


# ---- cusum_reference_value -----------------------------------------------
def test_cusum_reference_value():
    # one doc: clean token logit 0 (p=0.5), hallu token logit +1 (p=SIG1)
    ref, mu0, mu1 = L.cusum_reference_value([[0.5, SIG1]], [[0, 1]])
    assert mu0 == pytest.approx(0.0, abs=1e-6)
    assert mu1 == pytest.approx(1.0, abs=1e-6)
    assert ref == pytest.approx(0.5, abs=1e-6)


# ---- arl0_on_clean_stream -------------------------------------------------
def test_arl0_counts_upcrossings():
    # above=[F,F,T,F,T] -> 2 up-crossings over 5 tokens -> ARL0 = 2.5
    assert L.arl0_on_clean_stream([np.array([0, 0, 1, 0, 1])], 1.0) == pytest.approx(2.5)

def test_arl0_first_token_above_counts_once():
    # above=[T,F,F] -> 1 alarm over 3 tokens -> ARL0 = 3
    assert L.arl0_on_clean_stream([np.array([1, 0, 0])], 1.0) == pytest.approx(3.0)

def test_arl0_no_alarm_is_infinite():
    assert L.arl0_on_clean_stream([np.array([0, 0, 0])], 1.0) == float("inf")


# ---- edd_on_hallu ---------------------------------------------------------
def test_edd_detection_at_onset():
    edd, delay, recall = L.edd_on_hallu([np.array([0, 0, 1, 1])], [2], 1.0)
    assert delay == pytest.approx(0.0)
    assert edd == pytest.approx(0.0)
    assert recall == pytest.approx(1.0)

def test_edd_prechange_alarm_is_a_miss_censored():
    # crossing at t=0 < onset=2 -> miss, censored at len-theta = 4-2 = 2
    edd, delay, recall = L.edd_on_hallu([np.array([1, 0, 1, 1])], [2], 1.0)
    assert recall == pytest.approx(0.0)
    assert edd == pytest.approx(2.0)
    assert math.isnan(delay)


# ---- information_rate (properties) ---------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def test_information_rate_properties():
    # realistic scores: clean logits ~ N(-1.3, 1.2) (mean<0 but a positive tail,
    # like the real data mu0=-1.32), hallu logits ~ N(0.3, 1.0) (mu1=0.32). The
    # positive clean tail is what makes the Lundberg root exist.
    rng = np.random.RandomState(0)
    probs, labs = [], []
    for _ in range(60):
        l = np.zeros(40)
        lg = rng.normal(-1.3, 1.2, size=40)
        if rng.rand() < 0.5:
            l[20:] = 1.0
            lg[20:] = rng.normal(0.3, 1.0, size=20)
        probs.append(np.clip(_sigmoid(lg), 1e-6, 1 - 1e-6))
        labs.append(l)
    ref, _, _ = L.cusum_reference_value(probs, labs)
    r = L.information_rate(probs, labs, ref)
    assert r["delta1"] > 0          # post-change drift positive
    assert r["omega"] > 0           # Lundberg coefficient positive
    assert r["I"] == pytest.approx(r["omega"] * r["delta1"], rel=1e-9)
    assert r["drift0"] < 0          # pre-change drift negative

def test_information_rate_solves_lundberg_equation():
    # The defining property of the Lundberg coefficient omega: on the clean-token
    # increments Yc, E_0[exp(omega * Yc)] = 1. Verify the returned omega satisfies
    # it on the data (this checks the solver directly, free of any theoretical value).
    rng = np.random.RandomState(2)
    probs, labs = [], []
    for _ in range(60):
        l = np.zeros(40)
        lg = rng.normal(-1.3, 1.2, size=40)
        if rng.rand() < 0.5:
            l[20:] = 1.0
            lg[20:] = rng.normal(0.3, 1.0, size=20)
        probs.append(np.clip(_sigmoid(lg), 1e-6, 1 - 1e-6))
        labs.append(l)
    ref, _, _ = L.cusum_reference_value(probs, labs)
    r = L.information_rate(probs, labs, ref)

    # reconstruct clean increments Yc exactly as information_rate does
    clean = []
    for p, lb in zip(probs, labs):
        lo = np.log(p / (1 - p))
        clean.append(lo[np.asarray(lb) < 0.5] - ref)
    Yc = np.concatenate(clean)
    mgf_at_omega = float(np.mean(np.exp(r["omega"] * Yc)))
    assert mgf_at_omega == pytest.approx(1.0, abs=0.02)   # solver found the root
