"""
Evaluation for STRATUM models.

Reads a test set of {"prompt":..., "expected":...} and scores model outputs.
Ships three scorers; pick with --scorer:
  contains : expected string appears in output (lenient default)
  exact : output equals expected after normalization (strict)
  json_field: parse both as JSON, score field-by-field (for extraction)

See docs/07-evaluation.md for choosing and extending scorers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .data import load_jsonl, format_chat


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


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
    correct = sum(1 for k in keys if k in pred and _norm(pred[k]) == _norm(gold[k]))
    return correct / len(keys)


SCORERS = {"contains": score_contains, "exact": score_exact, "json_field": score_json_field}


def run_eval(model_dir: str, test_path: str, scorer: str = "contains",
             system: str | None = None, max_new_tokens: int = 256, verbose: bool = True):
    """Score a model on a test set. Returns the mean score in [0, 1]."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if scorer not in SCORERS:
        raise ValueError(f"Unknown scorer '{scorer}'. Choose from {list(SCORERS)}.")
    score_fn = SCORERS[scorer]

    rows = load_jsonl(test_path, required_keys=("prompt", "expected"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    total = 0.0
    for row in rows:
        text = format_chat(tokenizer, row["prompt"], response=None, system=system)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        answer = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        s = score_fn(answer, row["expected"])
        total += s
        if verbose:
            mark = "OK " if s >= 0.999 else ("~~ " if s > 0 else "XX ")
            print(f"{mark} score {s:.2f} expected: {str(row['expected'])[:36]:36} "
                  f"got: {answer.strip()[:36]}")

    mean = total / len(rows)
    print(f"\nScore ({scorer}): {mean:.1%} over {len(rows)} cases")
    return mean
