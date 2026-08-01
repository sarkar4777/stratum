"""
Evaluation for STRATUM models.

Reads a test set of {"prompt":..., "expected":...} and scores model outputs.
Ships four scorers, picked with --scorer:
  contains  : expected string appears in output (lenient default)
  exact     : output equals expected after normalization (strict)
  json_field: parse both as JSON, score field-by-field (for extraction)
  overlap   : word-overlap F1 (for free-text answers that paraphrase)

A test row may also carry a "skill" key. If any do, the report includes a
per-skill breakdown so you can see which stratum needs work after a merge.

run_eval returns a report dict and can write it as JSON (json_out) so a CI
job can gate on the score. See docs/08-evaluation.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .data import load_jsonl, format_chat, strip_think


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _as_number(v):
    """Parse a value as a number if possible: handles '1,499', '$88', '88.0'."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").lstrip("$")
    try:
        return float(s)
    except ValueError:
        return None


def _values_match(a, b) -> bool:
    """Compare two JSON field values, numerically when both parse as numbers.

    Without this, an expected 1499 would fail against a model that answered
    '1,499' - correct in substance, wrong only in formatting.
    """
    na, nb = _as_number(a), _as_number(b)
    if na is not None and nb is not None:
        return na == nb
    return _norm(a) == _norm(b)


def score_contains(output: str, expected) -> float:
    return float(_norm(expected) in _norm(output))


def score_exact(output: str, expected) -> float:
    return float(_norm(output) == _norm(expected))


def _extract_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def score_json_field(output: str, expected) -> float:
    pred = _extract_json(output)
    gold = expected if isinstance(expected, dict) else _extract_json(json.dumps(expected))
    if pred is None or not isinstance(gold, dict):
        return 0.0
    keys = [k for k in gold if gold[k] is not None]
    if not keys:
        return 0.0
    correct = sum(1 for k in keys if k in pred and _values_match(pred[k], gold[k]))
    return correct / len(keys)


_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset("""a an and are as at be by for from has have in is it its
of on or that the this to was were will with""".split())


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(_norm(text)) if w not in _STOPWORDS]


def score_overlap(output: str, expected) -> float:
    """Word-overlap F1 between the answer and the expected answer.

    The scorer for free-text answers, where `contains` is useless: a model
    that replies "A 100-metre blade made of glass fibre weighs 50 tonnes"
    to an expected "A 100-metre (330 ft) glass fiber blade weighs 50 tonnes"
    is right, and every substring test says zero.

    F1 balances two failure modes a single number has to catch: recall alone
    would reward an answer that recites the whole document, precision alone
    would reward a one-word guess. Common filler words are ignored so they
    cannot inflate the score. This is the standard measure for question
    answering, and like every automatic scorer it rewards saying the right
    words - pair it with human review for anything that matters.
    """
    pred = _content_words(output)
    gold = _content_words(str(expected))
    if not gold:
        return 0.0
    if not pred:
        return 0.0
    from collections import Counter
    shared = Counter(pred) & Counter(gold)
    n_shared = sum(shared.values())
    if n_shared == 0:
        return 0.0
    precision = n_shared / len(pred)
    recall = n_shared / len(gold)
    return 2 * precision * recall / (precision + recall)


SCORERS = {"contains": score_contains, "exact": score_exact,
           "json_field": score_json_field, "overlap": score_overlap}


def _generate(model, tokenizer, prompt, system, device, max_new_tokens):
    import torch
    text = format_chat(tokenizer, prompt, response=None, system=system)
    from .hf_utils import encode_for_generation
    ids = encode_for_generation(tokenizer, text, device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    # Thinking models may still open a reasoning block. Score only the answer.
    return strip_think(raw)


def run_eval(model_dir: str, test_path: str, scorer: str = "contains",
             system: str | None = None, max_new_tokens: int = 256,
             verbose: bool = True, json_out: str | None = None,
             baseline: str | None = None) -> dict:
    """Score a model on a test set.

    Returns a report dict: {"mean", "n", "scorer", "per_skill", "results"} plus,
    when baseline is given, "baseline_mean" - the same test run against another
    model (usually the untrained base) so the comparison is one command.
    """
    import torch

    if scorer not in SCORERS:
        raise ValueError(f"Unknown scorer '{scorer}'. Choose from {list(SCORERS)}.")
    score_fn = SCORERS[scorer]
    rows = load_jsonl(test_path, required_keys=("prompt", "expected"))
    from .hf_utils import pick_device
    device = pick_device()

    def eval_one_model(model_source):
        from .hf_utils import load_for_inference
        model, tokenizer = load_for_inference(model_source)
        model.to(device).eval()
        results = []
        for row in rows:
            answer = _generate(model, tokenizer, row["prompt"], system, device,
                               max_new_tokens)
            s = score_fn(answer, row["expected"])
            results.append({"prompt": row["prompt"], "expected": row["expected"],
                            "output": answer, "score": s,
                            "skill": row.get("skill")})
            if verbose:
                mark = "OK " if s >= 0.999 else ("~~ " if s > 0 else "XX ")
                print(f"{mark} score {s:.2f} expected: {str(row['expected'])[:36]:36} "
                      f"got: {answer.strip()[:36]}")
        # Release this model before a possible baseline model loads, so both
        # never sit in GPU memory at once.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    results = eval_one_model(model_dir)
    mean = sum(r["score"] for r in results) / len(results)

    per_skill = {}
    if any(r["skill"] for r in results):
        for r in results:
            per_skill.setdefault(r["skill"] or "(unlabeled)", []).append(r["score"])
        per_skill = {k: sum(v) / len(v) for k, v in sorted(per_skill.items())}

    report = {"model": model_dir, "test": test_path, "scorer": scorer,
              "n": len(results), "mean": mean, "per_skill": per_skill,
              "results": results}

    print(f"\nScore ({scorer}): {mean:.1%} over {len(results)} cases")
    for skill, s in per_skill.items():
        print(f"  {skill}: {s:.1%}")

    if baseline:
        print(f"\nBaseline run: {baseline}")
        base_results = eval_one_model(baseline)
        base_mean = sum(r["score"] for r in base_results) / len(base_results)
        report["baseline"] = baseline
        report["baseline_mean"] = base_mean
        print(f"Baseline score: {base_mean:.1%}  ->  your model: {mean:.1%} "
              f"({'+' if mean >= base_mean else ''}{(mean - base_mean):.1%})")

    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"Report written to {json_out}")

    return report
