# 9 - Full walkthrough

*Empty folder to working model, every command. Two ways: manual (learn each step) and recipe (one command). Follow on your own laptop.*

---

## Step 0 - understand it first (no GPU)

```bash
pip install numpy
python scripts/demo_concepts.py
```

You'll watch Muon flatten an update, an adapter hit its rank ceiling, and two strata fuse by addition - the three ideas you're about to use for real.

## Step 1 - install and check hardware

```bash
git clone https://github.com/sarkar4777/stratum.git
cd stratum
pip install -e .
pip install bitsandbytes # for 4-bit / QLoRA, if you have an NVIDIA GPU

stratum doctor
```

`doctor` reports your GPU/VRAM and recommends a base model. Use its recommendation for `--base`. When unsure, start smaller - it proves your pipeline before you scale.

Once you have a recipe (the one-command way at the end of this doc), `stratum plan recipe.yaml` goes further: it checks that *specific* build against this machine, suggests setting changes when it's tight, and writes a ready-to-run script for a rented GPU box when it doesn't fit at all (doc 10).

## Step 2 - prepare skill data

Each skill is a JSONL file, one `{"prompt","response"}` per line. See `examples/`. Quality rules:
- Be format-consistent - if you want JSON, make every response valid JSON.
- Cover the variety you expect at runtime, including hard cases.
- 100-500 pairs per skill is a fine start.

Hold back 10-20% as a test set with `expected` instead of `response`.

## Step 3 - train your first stratum

```bash
stratum train \
  --skill examples/extract.jsonl \
  --out strata/extract \
  --base Qwen/Qwen3-1.7B \
  --rank 16 \
  --epochs 3
```

You'll see `trainable% ~0.26`, then loss falling each epoch, then a saved stratum (a few MB) with its `stratum_card.json`.

## Step 4 - train a second stratum

Same command, different data. Different day is fine - strata don't know about each other:

```bash
stratum train --skill examples/classify.jsonl --out strata/classify \
  --base Qwen/Qwen3-1.7B --rank 16 --epochs 3
```

**Use the same `--base` for every stratum you'll merge.** STRATUM records it and refuses mismatched merges.

## Step 5 - fuse the strata

```bash
stratum merge strata/extract strata/classify --out models/my-slm --method linear
```

STRATUM checks the shared base, extracts each delta, combines them, applies onto a fresh base, and saves a standalone model. Emphasize a skill with weights:

```bash
stratum merge strata/extract strata/classify --out models/my-slm --weights 0.7 0.5
```

## Step 6 - evaluate

Score each skill with its matching scorer (doc 8 explains why one scorer can't judge two different skills):

```bash
stratum eval models/my-slm --test examples/test-extract.jsonl --scorer json_field
stratum eval models/my-slm --test examples/test-classify.jsonl --scorer exact
```

Prove the improvement against the untrained base in the same command:

```bash
stratum eval models/my-slm --test examples/test-extract.jsonl \
  --scorer json_field --baseline Qwen/Qwen3-1.7B
```

You can also score a stratum before ever merging it - point eval straight at the stratum folder and it attaches the adapter to its base for you. If a skill dropped after merging, try `--method ties`, then `--method dare`.

## Step 7 - use it

```bash
stratum chat models/my-slm
```

```
you: Extract the total from: 'Subtotal 40, tax 4, total 44'
stratum: {"total": 44}
you: Classify this ticket: 'my password reset link is broken'
stratum: account_access
```

One model, both skills, built in pieces on a laptop.

## Optional: distill from a bigger teacher for better quality

If you have access to a larger model, you can have it generate cleaner training
data (or teach your student directly). This often lifts quality noticeably. Two ways:

```bash
# Data distillation: a teacher writes the training pairs from seed inputs
stratum teacher-gen --seeds examples/seeds.txt \
  --instruction "Extract the invoice total as JSON." \
  --teacher hf --model Qwen/Qwen3-4B \
  --out examples/extract_distilled.jsonl
stratum train --skill examples/extract_distilled.jsonl --out strata/extract

# Logit distillation: student directly imitates a same-family teacher
stratum distill --skill examples/extract.jsonl --out strata/extract \
  --student Qwen/Qwen3-1.7B --teacher Qwen/Qwen3-4B
```

Either way you get an ordinary stratum that fuses like any other. Full explanation
in [doc 7](07-distillation.md).

## The one-command way: recipes

For repeatable, industry builds, put the whole thing in a YAML recipe and run it with `stratum stack`. See `examples/recipe.yaml`:

```yaml
base_model: Qwen/Qwen3-1.7B
optimizer: muon
system: "You are a precise assistant for invoice and ticket processing."
strata:
  - name: extract
    skill: examples/extract.jsonl
    out: strata/extract
    rank: 16
    epochs: 3
  - name: classify
    skill: examples/classify.jsonl
    out: strata/classify
    rank: 16
    epochs: 3
merge:
  method: linear
  weights: [1.0, 1.0]
output_model: models/my-slm
```

```bash
stratum stack examples/recipe.yaml
```

This trains every stratum and fuses them in one command - the reproducible build you'd check into a repo and run in CI. Change requirements? Edit the recipe, add a stratum, re-run.

Two properties make recipes safe to rely on. Every training setting (`lr`, `batch_size`, `grad_accum`, `max_len`, `seed`, `load_4bit`...) can be set recipe-wide and overridden per stratum, so the recipe expresses everything the CLI can. And the recipe is **validated before anything trains** - a misspelled key like `epoch:` is rejected with the list of valid keys, instead of being silently ignored while your build trains with a default you didn't choose.

A recipe can also end with **eval gates** (`evals:` - see `examples/recipe.yaml`): test sets with minimum scores that run right after the merge. A build that misses a bar fails, which turns the recipe into a self-verifying spec you can run unattended - locally, on a rented GPU box via `stratum plan --emit-remote`, or in CI. Docs 8 and 10 cover both ends of this.

## The whole thing, condensed

```bash
python scripts/demo_concepts.py # understand
stratum doctor # size
stratum train --skill A.jsonl --out strata/a # stratum 1
stratum train --skill B.jsonl --out strata/b # stratum 2
stratum merge strata/a strata/b --out models/mine # fuse
stratum eval models/mine --test test.jsonl # measure
stratum chat models/mine # use
# or, all at once:
stratum stack recipe.yaml
```

Next: [scaling up - bigger bases, more strata, real deployment, and industry patterns ->](10-scaling-and-production.md)
