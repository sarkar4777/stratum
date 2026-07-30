# 3 - LoRA and adapters

*How a stratum trains under 1% of a model and still teaches real skills - the mechanism, the one knob that matters, and how it maps to STRATUM's code.*

---

## The problem

Full fine-tuning is too memory-hungry (doc 1) because you need gradients and optimizer notes for every dial. The fix: freeze the model, train only a small add-on called an **adapter**. The popular method is **LoRA** - *Low-Rank Adaptation*. Let's unpack that name properly, no gaps.

## What's inside a model

Most of a model's work is multiplying data by big grids of numbers - **weight matrices**. A typical one might be 2,560 x 2,560, about 6.5 million numbers, and there are hundreds of them. Normal fine-tuning adjusts all 6.5 million per grid. That's the expense.

## The LoRA insight

When you *specialize* a model, the change each grid needs is **simple** - not random noise across all 6.5 million numbers, but a structured, low-complexity adjustment. And a simple grid can be reconstructed from two much smaller grids multiplied together.

**Analogy.** Picture a 1000x1000 spreadsheet where every cell equals (its row number) x (its column number). A million cells - but you only need the 1000 row numbers and 1000 column numbers to regenerate any cell. A million values described by two thousand. That grid is "low rank": a small description reconstructs it.

LoRA applies this to the *change* it learns:
- Freeze the original grid `W`. Never touch it.
- Add two small grids: `A` (skinny, say 16 rows) and `B` (skinny, 16 columns).
- Their product `B x A` has the same shape as `W`, but is built from far fewer numbers.
- The model computes `W + (B x A)` - frozen original plus small learned adjustment.

Only `A` and `B` train. Where `W` had 6.5 million numbers, `A`+`B` might have 80,000 - about **1.25%**.

## Rank: the one knob that matters

The skinniness of `A` and `B` - how many rows/columns - is the **rank**, `r`. It's your main dial.

- **Low rank** (8-16): cheap, but the adjustment can only be so complex. Good for **style, format, simple behavior**.
- **Higher rank** (32-64): more capacity, more memory. Needed for **real domain knowledge**.

There's a hard limit worth seeing. An adapter can only represent an adjustment as complex as its rank. Ask a rank-4 adapter to learn something needing rank-20 complexity and it gets partway, then **plateaus forever** - not slow, a ceiling. STRATUM's demo shows this with real numbers:

```
This skill needs a rank-6 adjustment. Error before training: 1.630

 adapter rank final error gap closed
            1 0.9201 43.6% <- stuck
            2 0.4879 70.1% <- stuck
            4 0.1299 92.0% <- stuck
            6 0.0000 100.0%
           12 0.0000 100.0%
```

Run it yourself: `python scripts/demo_concepts.py`. **Rule:** rank 16 for style/format strata, 32-64 for knowledge-heavy strata. Using rank 8 for a knowledge skill is the classic mistake - it half-learns and you wrongly blame the method.

## Why an adapter is a perfect tile

An adapter is **tiny** (megabytes), **separate** (its own file beside the frozen model), and **additive** (its effect is *added*: `W + BxA`). Those three properties are exactly what make it a mergeable stratum - you make one on Monday, another Tuesday, keep them in a folder, fuse whenever. All trained against the same frozen base, so all compatible. Doc 5 is the fusing.

## Merge vs keep-separate

Two ways to use a trained adapter:
1. **Keep separate** - load frozen base + adapter at runtime, and the model computes `W + BxA` live. Flexible: swap adapters freely.
2. **Merge in** - do the `W + BxA` math once, save a standalone model. Simpler, slightly faster (no separate adapter to apply).

STRATUM fuses strata together, then applies the result onto the base, producing one clean model - but the strata exist as separate files first, which is what enables piece-by-piece building.

## QLoRA in one line

**QLoRA** = LoRA + compressing the frozen base to 4 bits (doc 1's bonus lever). Since the base is only read, compressing it costs almost no quality, and it's what fits a 4B model's training on 8 GB. STRATUM turns this on by default when you have a GPU (`--no-4bit` disables it).

Credit where due: LoRA is from Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021), and QLoRA from Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023). STRATUM builds on Hugging Face's PEFT library, which implements both.

## In the code

STRATUM uses Hugging Face PEFT for LoRA. In `stratum/train.py`:

```python
lora_cfg = LoraConfig(
    r=rank, # your --rank
    lora_alpha=rank * 2, # scaling, convention is 2x rank
    lora_dropout=0.05,
    target_modules="all-linear", # adapt every linear grid
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_cfg)
```

`target_modules="all-linear"` puts an adapter on every linear grid, which outperforms the old habit of adapting only some. `lora_alpha` is a scaling factor (`BxA` is multiplied by `alpha/r`), and keeping it at `2xr` means changing rank doesn't silently change your effective learning rate.

## What you now know

- An **adapter** freezes the model and trains a small add-on.
- **LoRA** builds it as two small grids whose product `BxA` is the adjustment - ~1% the size.
- **Rank** sets capacity - too low and it plateaus. 16 for style, 32-64 for knowledge.
- Adapters are tiny, separate, additive - i.e. **mergeable strata**.
- **QLoRA** compresses the frozen base to fit a laptop.

Next: [Muon - the optimizer that trains strata faster and lighter than AdamW ->](04-muon-explained.md)
