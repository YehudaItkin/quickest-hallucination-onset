#!/usr/bin/env python3
"""CPU-only tests for the consistency block's pure alignment logic.

These cover the correctness-critical invariant: the per-token consistency array
must have exactly len(token_labels) rows and broadcast each sentence's score onto
its own tokens, matching token_features._detect_sentence_boundaries. No torch /
transformers / run_extended needed -- runs anywhere.

    python test_consistency_alignment.py        # or: pytest test_consistency_alignment.py
"""
from server_consistency_features import (
    sentence_spans,
    sentence_text,
    broadcast_to_tokens,
    ngram_inconsistency,
    selfcheck_nli_from_probs,
)


def _toks(words):
    return [{"word": w, "is_hallucination": 0} for w in words]


def test_sentence_spans_tile_without_gaps():
    tl = _toks(["The", "cat", ".", "It", "ran", "!", "Now", "?"])
    spans = sentence_spans(tl)
    # boundaries open after '.', '!', '?' (not the trailing one)
    assert spans == [(0, 3), (3, 6), (6, 8)], spans
    # spans must tile [0, n) with no gaps or overlaps
    flat = [t for s, e in spans for t in range(s, e)]
    assert flat == list(range(len(tl)))


def test_sentence_spans_single_sentence():
    tl = _toks(["no", "punctuation", "here"])
    assert sentence_spans(tl) == [(0, 3)]


def test_sentence_spans_trailing_period_no_empty_span():
    tl = _toks(["End", "."])
    # '.' is the last token (i+1 == n) so it does NOT open an empty trailing span
    assert sentence_spans(tl) == [(0, 2)]


def test_sentence_text_reconstruction():
    tl = _toks(["The", "cat", "."])
    assert sentence_text(tl, (0, 3)) == "The cat ."


def test_broadcast_length_and_values():
    tl = _toks(["The", "cat", ".", "It", "ran", "."])  # spans [(0,3),(3,6)]
    per_sentence = [[0.9, 0.1], [0.2, 0.8]]
    out = broadcast_to_tokens(tl, per_sentence)
    assert len(out) == len(tl)              # the invariant the divergence script asserts
    assert out[0] == [0.9, 0.1] and out[2] == [0.9, 0.1]
    assert out[3] == [0.2, 0.8] and out[5] == [0.2, 0.8]


def test_broadcast_rejects_wrong_sentence_count():
    tl = _toks(["a", ".", "b", "."])
    try:
        broadcast_to_tokens(tl, [[1.0]])   # 1 score, 2 spans
        raise AssertionError("should have raised on mismatched sentence count")
    except AssertionError as e:
        assert "sentence" in str(e)


def test_ngram_inconsistency_bounds():
    sent = "Paris is the capital of France"
    assert ngram_inconsistency(sent, ["Paris capital France"]) < 0.5   # well supported
    assert ngram_inconsistency(sent, ["totally unrelated text"]) > 0.5  # unsupported
    assert ngram_inconsistency("the of a", ["anything"]) == 0.0         # no content words


def test_selfcheck_nli_mean():
    assert selfcheck_nli_from_probs([0.0, 1.0]) == 0.5
    assert selfcheck_nli_from_probs([]) == 0.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
