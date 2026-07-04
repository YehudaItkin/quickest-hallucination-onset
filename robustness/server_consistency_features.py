#!/usr/bin/env python3
"""Black-box multi-sample CONSISTENCY features for hallucination onset detection.

Motivation. The CUSUM paper measures a gap between the Lorden floor (~1.3 tokens)
and the achieved delay (11-13 tokens), and attributes most of it to a realized
information rate I(g_hat) ~ D/4.5, calling the 4.5x deficit "close to irreducible
*for these features*". The data-processing inequality says no transform of the
existing 33-d Text+NLI+LM stream can raise D, so we add a genuinely NEW black-box
measurement: does the generator say the same thing if you resample it K times?
Hallucinated spans are unstable across resamples (epistemic uncertainty), faithful
ones are not. This is orthogonal to the aleatoric LM-entropy block we already have.

What this script computes (per token, aligned 1:1 with token_labels):
  - selfcheck_nli   : mean over K samples of the normalized contradiction prob
                      (SelfCheckGPT-NLI, Manakul et al. 2023) of the token's sentence
                      against each resample. Higher = more inconsistent.
  - selfcheck_ngram : 1 - mean unigram support of the sentence's content words in
                      the resamples (cheap lexical inconsistency, orthogonal proxy).

Sentence segmentation MIRRORS token_features._detect_sentence_boundaries (split
after a token whose word.rstrip() is in {'.', '!', '?'}), so the broadcast lands
on exactly the token rows that load_and_enrich_all / assemble_features produce.

Black-box: needs only sampling access to the generator + an off-the-shelf NLI head.
Resampling MUST use the SAME generator that produced the response, so we filter by
example["model"] and load the matching open-weight checkpoint.

Run on the GPU server in ~/hallucination_exp (needs run_extended + token_features
importable, plus transformers/torch and the generator weights). Heavy imports are
lazy so the pure-logic functions below can be unit-tested on CPU without torch.

Usage:
  python server_consistency_features.py --list-models          # inspect model field
  python server_consistency_features.py --models llama-2-7b-chat --split train --k 5
"""
import argparse
import json
import re

# ---------------------------------------------------------------------------
# Pure logic (no torch / transformers) -- unit-tested by test_consistency_alignment.py
# ---------------------------------------------------------------------------

SENT_END = (".", "!", "?")


def sentence_spans(token_labels):
    """Return [(start, end)] token-index ranges per sentence.

    Mirrors token_features._detect_sentence_boundaries: a boundary opens right
    after any token whose word.rstrip() is sentence-final punctuation. The spans
    tile [0, len(token_labels)) with no gaps, so every token belongs to exactly
    one sentence.
    """
    n = len(token_labels)
    starts = [0]
    for i, tok in enumerate(token_labels):
        if tok["word"].rstrip() in SENT_END and i + 1 < n:
            starts.append(i + 1)
    spans = []
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else n
        spans.append((s, e))
    return spans


def sentence_text(token_labels, span):
    """Reconstruct a sentence's surface text from its token words."""
    s, e = span
    return " ".join(tok["word"] for tok in token_labels[s:e]).strip()


def broadcast_to_tokens(token_labels, per_sentence):
    """Broadcast per-sentence feature vectors to a (n_tokens, n_dims) matrix.

    per_sentence is a list aligned with sentence_spans(token_labels); each entry is
    a list of feature values. Returns a list-of-lists of length len(token_labels).
    """
    spans = sentence_spans(token_labels)
    assert len(per_sentence) == len(spans), (
        f"{len(per_sentence)} sentence scores vs {len(spans)} spans"
    )
    n = len(token_labels)
    n_dims = len(per_sentence[0]) if per_sentence else 0
    out = [[0.0] * n_dims for _ in range(n)]
    for (s, e), vec in zip(spans, per_sentence):
        for t in range(s, e):
            out[t] = list(vec)
    return out


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "by", "with", "as", "at", "it", "this", "that", "from",
}


def _content_words(text):
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP]


def ngram_inconsistency(sentence, samples):
    """1 - mean fraction of the sentence's content words that appear in a sample.

    A purely lexical, model-free inconsistency proxy that is orthogonal to the NLI
    score. Returns 0.0 (fully supported) when the sentence has no content words.
    """
    words = _content_words(sentence)
    if not words:
        return 0.0
    sample_vocabs = [set(_content_words(s)) for s in samples]
    if not sample_vocabs:
        return 0.0
    supports = []
    for vocab in sample_vocabs:
        hit = sum(1 for w in words if w in vocab)
        supports.append(hit / len(words))
    mean_support = sum(supports) / len(supports)
    return float(1.0 - mean_support)


def selfcheck_nli_from_probs(contradiction_probs):
    """SelfCheckGPT-NLI sentence score = mean normalized contradiction prob over K
    samples. contradiction_probs is the per-sample P(contra)/(P(entail)+P(contra))."""
    if not contradiction_probs:
        return 0.0
    return float(sum(contradiction_probs) / len(contradiction_probs))


# ---------------------------------------------------------------------------
# Model plumbing (lazy heavy imports inside the functions)
# ---------------------------------------------------------------------------

# RAGTruth model field (lowercased, matched by substring) -> HF checkpoint.
MODEL_HF_MAP = {
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
}
NLI_MODEL = "microsoft/deberta-large-mnli"  # MNLI head (cached on server), SelfCheck-NLI style


def resolve_hf_id(model_field, override=None):
    if override:
        return override
    key = model_field.lower().replace("_", "-")
    for sub, hf in MODEL_HF_MAP.items():
        if sub in key:
            return hf
    raise ValueError(
        f"No HF checkpoint mapped for model={model_field!r}; pass --hf-id "
        f"(open-weight only; GPT models need an API path)."
    )


def load_generator(hf_id, load_4bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"device_map": "auto"}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kw["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kw)
    model.eval()
    return tok, model


def load_nli():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    # locate entailment / contradiction logit indices robustly
    label2id = {k.lower(): v for k, v in model.config.label2id.items()}
    ent = label2id.get("entailment")
    con = label2id.get("contradiction")
    assert ent is not None and con is not None, model.config.label2id
    return tok, model, ent, con


def sample_continuations(tok, model, context, k, temperature, max_new_tokens,
                         max_prompt=2048, gen_batch=1):
    """K stochastic continuations. Generated in chunks of gen_batch to cap the KV
    cache (5 long sequences at once can overflow a shared GPU); prompt truncated to
    max_prompt tokens for the same reason."""
    import torch

    msgs = [{"role": "user", "content": context}]
    try:
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = context
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_prompt)
    enc = {kk: vv.to(model.device) for kk, vv in enc.items()}
    plen = enc["input_ids"].shape[1]
    samples, remaining = [], k
    with torch.no_grad():
        while remaining > 0:
            b = min(gen_batch, remaining)
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                num_return_sequences=b,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
            samples.extend(
                tok.decode(g, skip_special_tokens=True).strip() for g in out[:, plen:]
            )
            remaining -= b
    return samples


def nli_contradiction_prob(nli_tok, nli_model, ent_idx, con_idx, premise, hypothesis):
    """P(contra)/(P(entail)+P(contra)) for (premise -> hypothesis), SelfCheck-NLI."""
    import torch

    enc = nli_tok(
        premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
    )
    enc = {kk: vv.to(nli_model.device) for kk, vv in enc.items()}
    with torch.no_grad():
        logits = nli_model(**enc).logits[0]
    pair = torch.softmax(logits[[ent_idx, con_idx]], dim=-1)
    return float(pair[1])  # contradiction share over {entail, contra}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_examples(split):
    """Examples in the canonical loader order (so ids align with load_and_enrich_all)."""
    from run_extended import load_base_features

    _, _, examples = load_base_features(split)
    return examples


def compute_for_example(ex, gen, nli, k, temperature, max_prompt=2048, gen_batch=1):
    tok, model = gen
    nli_tok, nli_model, ent_idx, con_idx = nli
    token_labels = ex["token_labels"]
    spans = sentence_spans(token_labels)
    target_len = max(64, min(512, int(1.5 * len(token_labels))))
    samples = sample_continuations(tok, model, ex["context"], k, temperature,
                                   target_len, max_prompt, gen_batch)

    per_sentence = []
    for span in spans:
        sent = sentence_text(token_labels, span)
        if not sent:
            per_sentence.append([0.0, 0.0])
            continue
        contra = [
            nli_contradiction_prob(nli_tok, nli_model, ent_idx, con_idx, s, sent)
            for s in samples if s
        ]
        nli_score = selfcheck_nli_from_probs(contra)
        ngram_score = ngram_inconsistency(sent, samples)
        per_sentence.append([nli_score, ngram_score])
    return broadcast_to_tokens(token_labels, per_sentence)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--models", nargs="*", default=None,
                    help="substring filter on example['model']; open-weight only")
    ap.add_argument("--hf-id", default=None, help="override HF checkpoint")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0, help="cap #examples (0=all)")
    ap.add_argument("--load-4bit", action="store_true",
                    help="4-bit (bitsandbytes) so the generator fits a busy GPU")
    ap.add_argument("--max-prompt", type=int, default=2048,
                    help="truncate the RAG context to this many tokens (KV-cache cap)")
    ap.add_argument("--gen-batch", type=int, default=1,
                    help="continuations generated at once (raise on a free GPU)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    examples = load_examples(args.split)

    if args.list_models:
        from collections import Counter
        c = Counter(e["model"] for e in examples)
        print(f"{args.split}: {len(examples)} examples")
        for m, n in c.most_common():
            print(f"  {m:32s} {n:6d}")
        return

    sel = examples
    if args.models:
        keys = [m.lower() for m in args.models]
        sel = [e for e in examples
               if any(kk in e["model"].lower().replace("_", "-") for kk in keys)]
    if args.limit:
        sel = sel[:args.limit]
    if not sel:
        raise SystemExit("no examples matched --models filter")

    model_fields = sorted({e["model"] for e in sel})
    print(f"Selected {len(sel)} examples across models: {model_fields}", flush=True)

    # All selected examples must share one generator (so resampling is from the
    # SAME model). Run per --models value if you need several.
    hf_id = resolve_hf_id(model_fields[0], args.hf_id)
    if len(model_fields) > 1 and not args.hf_id:
        raise SystemExit(
            f"selection spans {model_fields}; run once per model (resampling must "
            f"use the same generator) or pass --hf-id explicitly."
        )
    print(f"Generator: {hf_id} | NLI: {NLI_MODEL} | K={args.k} T={args.temperature}",
          flush=True)

    gen = load_generator(hf_id, args.load_4bit)
    nli = load_nli()

    out_path = args.out or f"consistency_feats_{args.split}.json"
    feats = {}
    meta = {"split": args.split, "k": args.k, "temperature": args.temperature,
            "hf_id": hf_id, "nli_model": NLI_MODEL, "models": model_fields,
            "dims": ["selfcheck_nli", "selfcheck_ngram"]}
    for n, ex in enumerate(sel):
        feats[str(ex["id"])] = compute_for_example(
            ex, gen, nli, args.k, args.temperature, args.max_prompt, args.gen_batch)
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(sel)}", flush=True)
            json.dump({"meta": meta, "feats": feats}, open(out_path, "w"))
    json.dump({"meta": meta, "feats": feats}, open(out_path, "w"))
    print(f"Saved {len(feats)} examples -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
