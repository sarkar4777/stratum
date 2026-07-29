"""
Data loading and formatting for STRATUM.

A skill file is JSONL: one JSON object per line with "prompt" and "response"
(training) or "prompt" and "expected" (eval). This module:
  - loads and validates those files with clear errors
  - formats each pair into the base model's chat template
  - builds batches with a proper loss mask (train only on the response tokens)

The loss mask is the single most important correctness detail in fine-tuning:
we must compute loss ONLY on the assistant's response, never on the prompt.
See docs/06-training.md for why.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def load_jsonl(path: str, required_keys: tuple[str, ...]) -> list[dict]:
    """Load and validate a JSONL file. Raises with line numbers on bad rows."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    rows = []
    for i, line in enumerate(p.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} line {i}: invalid JSON ({e}).")
        missing = [k for k in required_keys if k not in obj]
        if missing:
            raise ValueError(f"{path} line {i}: missing key(s) {missing}. "
                             f"Each line needs {list(required_keys)}.")
        rows.append(obj)

    if not rows:
        raise ValueError(f"{path} contains no usable rows.")
    return rows


def format_chat(tokenizer, prompt: str, response: str | None = None,
                system: str | None = None) -> str:
    """Render a prompt (and optional response) into the model's chat format.

    If response is None, adds the generation prompt (for inference). Falls back
    to a simple template if the tokenizer lacks a chat template.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    if response is not None:
        messages.append({"role": "assistant", "content": response})

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=(response is None),
        )
    except Exception:
        text = ""
        if system:
            text += f"System: {system}\n"
        text += f"User: {prompt}\nAssistant:"
        if response is not None:
            text += f" {response}{tokenizer.eos_token}"
        return text


def build_example(tokenizer, prompt, response, system, max_len):
    """Tokenize one (prompt, response) pair and build a loss mask.

    Returns (input_ids, labels) where labels are -100 on the prompt portion so
    loss is computed only on the response. This is done by tokenizing the
    prompt-only text first to find where the response begins.
    """
    prompt_text = format_chat(tokenizer, prompt, response=None, system=system)
    full_text = format_chat(tokenizer, prompt, response=response, system=system)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False,
                         truncation=True, max_length=max_len)["input_ids"]

    labels = list(full_ids)
    # Mask everything up to where the response starts.
    mask_until = min(len(prompt_ids), len(full_ids))
    for i in range(mask_until):
        labels[i] = -100

    return full_ids, labels


def make_batches(tokenizer, rows, system, max_len, batch_size):
    """Yield padded batches of (input_ids, attention_mask, labels) tensors."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    examples = [build_example(tokenizer, r["prompt"], r["response"], system, max_len)
                for r in rows]

    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        maxlen = max(len(ids) for ids, _ in chunk)

        input_ids, attn, labels = [], [], []
        for ids, labs in chunk:
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(labs + [-100] * pad)

        yield (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
