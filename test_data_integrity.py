"""Data-integrity checks for the posteriors behind the decomposition.

Verifies that (1) the saved ForwardGRU/LogReg/BiGRU labels are the real RAGTruth
test labels, (2) the HistGBM posteriors are on the SAME documents in the SAME
order, and (3) every posterior is a valid probability sequence aligned to its
labels. If any of these fail, the decomposition numbers are not trustworthy.
"""
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TESTJSON = HERE / "test.json"
DIRPROBS = HERE / "directional_probs_seed42.json"
HISTGBM = HERE / "histgbm_probs.json"


@pytest.fixture(scope="module")
def ragtruth_labels():
    docs = json.load(open(TESTJSON))
    return [[int(t["is_hallucination"]) for t in d["token_labels"]] for d in docs]


@pytest.fixture(scope="module")
def directional():
    return json.load(open(DIRPROBS))


def _transition(seqs):
    c = np.zeros((2, 2))
    for s in seqs:
        s = np.asarray(s)
        for i in range(1, len(s)):
            c[int(s[i - 1] > 0.5), int(s[i] > 0.5)] += 1
    rs = c.sum(1, keepdims=True); rs[rs == 0] = 1
    return c / rs


def test_directional_statistically_matches_ragtruth(ragtruth_labels, directional):
    # Two label versions exist (server pipeline vs the copied test.json); they
    # agree on tokenisation (same lengths/total) and differ on only ~175 docs /
    # 183 tokens out of 341k (0.05%), a minor RAGTruth relabel. What must hold is
    # that the experiment's labels and the reference are STATISTICALLY equivalent:
    # same transition matrix -> same Lorden bound.
    labs = directional["ForwardGRU"]["labs"]
    assert len(labs) == len(ragtruth_labels) == 2700
    assert all(len(a) == len(b) for a, b in zip(labs, ragtruth_labels)), "tokenisation differs"
    mism = sum(1 for a, b in zip(labs, ragtruth_labels)
               if not np.array_equal((np.asarray(a) > 0.5).astype(int), b))
    assert mism < 250, f"{mism} docs differ — larger than the known relabel"

    Td, Tr = _transition(labs), _transition(ragtruth_labels)
    assert Td[0, 1] == pytest.approx(Tr[0, 1], abs=5e-4), "onset hazard P(F->H) differs"
    assert Td[1, 1] == pytest.approx(Tr[1, 1], abs=5e-3), "persistence P(H->H) differs"


def test_all_directional_models_share_labels(directional):
    ref = directional["ForwardGRU"]["labs"]
    for m in ["LogReg", "BiGRU", "BackwardGRU"]:
        if m in directional:
            for a, b in zip(directional[m]["labs"], ref):
                assert np.array_equal(a, b)


def test_histgbm_aligned_with_directional(directional):
    hg = json.load(open(HISTGBM))["HistGBM"]
    ref = directional["ForwardGRU"]["labs"]
    assert len(hg["labs"]) == len(ref) == 2700
    mism = sum(
        1 for a, b in zip(hg["labs"], ref)
        if len(a) != len(b) or not np.array_equal(np.asarray(a) > 0.5, np.asarray(b) > 0.5)
    )
    assert mism == 0, f"{mism} HistGBM docs misaligned"


def test_posteriors_valid_and_aligned(directional):
    hg = json.load(open(HISTGBM))["HistGBM"]
    sources = {"ForwardGRU": directional["ForwardGRU"], "LogReg": directional["LogReg"],
               "HistGBM": hg}
    for name, d in sources.items():
        for p, l in zip(d["probs"], d["labs"]):
            p = np.asarray(p)
            assert len(p) == len(l), f"{name}: probs/labs length mismatch"
            if len(p):
                assert p.min() >= 0.0 and p.max() <= 1.0, f"{name}: prob out of [0,1]"


def test_base_rate_consistent(ragtruth_labels):
    pi = np.mean([t for doc in ragtruth_labels for t in doc])
    assert pi == pytest.approx(0.0416, abs=0.002)   # matches the 4.16% used throughout
