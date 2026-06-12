"""Tests for the nonlinear-baseline delay experiment (TDD).

delay_at_arl0(probs, labs, mode, target_arl0) wraps the run_learned_cusum
detectors: split clean/hallu by onset, build threshold- or CUSUM-paths, sweep,
and read the operating point at a matched ARL0. These tests pin its behaviour on
deterministic synthetic data where the answer is known by hand.
"""
import numpy as np
import pytest

from nonlinear_baseline import delay_at_arl0


def _synthetic(onset=5, length=20, n=5, p_lo=0.02, p_hi=0.98):
    """n clean docs (label 0, low score) + n hallu docs (label 1 from `onset`,
    score jumps low->high exactly at the onset)."""
    probs, labs = [], []
    for _ in range(n):                       # clean
        probs.append(np.full(length, p_lo)); labs.append(np.zeros(length))
    for _ in range(n):                       # hallu
        pr = np.full(length, p_lo); pr[onset:] = p_hi
        lb = np.zeros(length); lb[onset:] = 1.0
        probs.append(pr); labs.append(lb)
    return probs, labs


def test_threshold_detects_at_onset_with_no_false_alarms():
    probs, labs = _synthetic(onset=5)
    op = delay_at_arl0(probs, labs, mode="threshold", target_arl0=50)
    assert op is not None
    assert op["recall"] == 1.0            # every hallu doc caught
    assert op["delay"] == 0.0             # fires exactly at the onset token
    assert op["arl0"] == float("inf")     # clean docs never cross a mid threshold


def test_later_onset_gives_zero_delay_too():
    # delay is measured from the onset, so a later onset still yields delay 0 here
    probs, labs = _synthetic(onset=12)
    op = delay_at_arl0(probs, labs, mode="threshold", target_arl0=50)
    assert op["delay"] == 0.0
    assert op["recall"] == 1.0


def test_cusum_mode_runs_and_detects():
    probs, labs = _synthetic(onset=5)
    op = delay_at_arl0(probs, labs, mode="cusum", target_arl0=50)
    assert op is not None
    assert op["recall"] >= 0.8            # CUSUM also catches the clean onset jump
    assert op["delay"] < 5.0              # within a few tokens of the onset


def test_unknown_mode_raises():
    probs, labs = _synthetic()
    with pytest.raises(ValueError):
        delay_at_arl0(probs, labs, mode="bogus", target_arl0=50)
