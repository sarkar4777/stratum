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

One extra wrinkle handled here: "thinking" models (Qwen3 and friends) have
chat templates that insert <think> reasoning blocks. We render templates with
thinking disabled so training targets and inference prompts line up, and
strip_think() removes any think blocks a model still emits.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch


def file_sha256(path: str) -> str:
    """Fingerprint of a data file, recorded in stratum cards.

    An auditor (or you, six months later) can prove which exact data a
    stratum was trained on by hashing the file again and comparing.
    """
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str, required_keys: tuple[str, ...]) -> list[dict]:
    """Load and validate a JSONL file. Raises with line numbers on bad rows."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    rows = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
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


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks a thinking model may emit.

    Qwen3 and similar models reason inside think tags before answering. For
    scoring, chat display, and teacher-generated training data we want only
    the answer. An unclosed <think> (generation ran out of tokens mid-thought)
    is cut from the tag onward.
    """
    text = _THINK_RE.sub("", text)
    open_tag = text.find("<think>")
    if open_tag != -1:
        text = text[:open_tag]
    return text.strip()


def format_messages(tokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    """Render a message list with the tokenizer's chat template.

    Thinking is disabled (enable_thinking=False) so models like Qwen3 answer
    directly, and templates that don't know the flag simply ignore it. Falls back
    to a plain format if the tokenizer has no chat template.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except Exception:
        text = ""
        for m in messages:
            role = {"system": "System", "user": "User", "assistant": "Assistant"}[m["role"]]
            text += f"{role}: {m['content']}"
            if m["role"] == "assistant":
                text += tokenizer.eos_token or ""
            text += "\n"
        if add_generation_prompt:
            text += "Assistant:"
        return text


def format_chat(tokenizer, prompt: str, response: str | None = None,
                system: str | None = None) -> str:
    """Render a prompt (and optional response) into the model's chat format.

    If response is None, adds the generation prompt (for inference).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return format_messages(tokenizer, messages, add_generation_prompt=(response is None))


def build_example(tokenizer, prompt, response, system, max_len):
    """Tokenize one (prompt, response) pair and build a loss mask.

    Returns (input_ids, labels) where labels are -100 on the prompt portion so
    loss is computed only on the response.

    The boundary must be exact, so the prompt and response are tokenized
    separately and concatenated - no guessing where the response starts inside
    a jointly tokenized string. When the full-conversation rendering doesn't
    begin with the generation prompt (thinking templates insert an empty think
    block there), we train on generation_prompt + response + eos instead, so
    training matches exactly what the model sees at inference.
    """
    prompt_text = format_chat(tokenizer, prompt, response=None, system=system)
    full_text = format_chat(tokenizer, prompt, response=response, system=system)

    if full_text.startswith(prompt_text):
        response_text = full_text[len(prompt_text):]
    else:
        response_text = response + (tokenizer.eos_token or "")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + response_ids)[:max_len]
    n_response = len(input_ids) - len(prompt_ids)
    if n_response <= 0:
        raise ValueError(
            f"A training row has no response tokens within --max-len={max_len}: "
            f"the prompt alone is {len(prompt_ids)} tokens. Training on it would "
            f"teach nothing. Raise --max-len or shorten the prompt.\n"
            f"Prompt starts: {prompt[:80]!r}"
        )
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    return input_ids, labels


def check_training_data(tokenizer, rows, system, max_len, verbose=True) -> dict:
    """Look at the data before training and report what will go wrong.

    Two failure modes cost real GPU time and are invisible until you evaluate:

    Very short responses. If most answers are a handful of tokens, the
    end-of-sequence token dominates the loss and the model can learn to emit
    it immediately - producing a stratum that answers every prompt with
    nothing. Terse extraction data is exactly where this bites.

    Too few pairs. Under about 50 examples a stratum memorizes rather than
    generalizes, and the eval numbers that follow mean very little.

    Returns the statistics and prints warnings with the fix. Never raises -
    the user decides whether to proceed.
    """
    lengths = []
    for r in rows:
        ids, labels = build_example(tokenizer, r["prompt"], r["response"],
                                    system, max_len)
        lengths.append(sum(1 for l in labels if l != -100))
    lengths.sort()
    n = len(lengths)
    median = lengths[n // 2]
    stats = {"pairs": n, "median_response_tokens": median,
             "shortest_response_tokens": lengths[0],
             "longest_response_tokens": lengths[-1]}

    if verbose:
        print(f"Training data: {n} pairs, response length "
              f"{lengths[0]}-{lengths[-1]} tokens (median {median})")
        if median <= 8:
            print(" WARNING: responses are very short. The end-of-sequence "
                  "token can dominate\n"
                  "  the loss and leave a stratum that answers with nothing at "
                  "all. If that\n"
                  "  happens: lower --lr (try 5e-3), use --epochs 1-2, or write "
                  "answers as\n"
                  "  short sentences rather than bare values. Always check with "
                  "`stratum eval`.")
        if n < 50:
            print(f" NOTE: {n} pairs is below the ~100-500 a skill wants "
                  f"(doc 10). Fine for\n"
                  f"  proving the pipeline, thin for a real skill.")
    return stats


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
