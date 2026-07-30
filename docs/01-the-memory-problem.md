# 1 - The memory problem

*Why normal training needs an expensive machine, and exactly which levers shrink it to laptop size. This is the "why STRATUM works" doc.*

---

## The constraint is memory, not speed

Slow training is annoying but survivable - you wait. The real wall is **VRAM** (the memory on your graphics card). If a training step doesn't fit in VRAM, it doesn't run at all. Hard stop. So the whole game is shrinking the memory a training step needs.

## Where the memory goes: full training a 4B model

Each number is stored in some number of bytes, and a common training choice is 2 bytes. Full training - adjusting every dial the normal way - needs, per the standard optimizer AdamW:

| What's stored | Why | Memory (4B) |
|---|---|---|
| Model weights | the dials | 8 GB |
| Gradients | nudge direction per dial | 8 GB |
| Optimizer note 1 (momentum) | AdamW remembers recent nudges | 16 GB |
| Optimizer note 2 (variance) | AdamW remembers nudge sizes | 16 GB |
| High-precision weight copy | apply tiny nudges without rounding | 16 GB |
| **Total** | | **~64 GB** + working space |

A high-end **laptop** GPU has 8 GB. A strong desktop card has 24 GB. A top data-center GPU has 80 GB and *barely* fits this. That table is the entire problem.

**Notice the biggest cost:** the three optimizer-related rows (momentum + variance + high-precision copy) are 48 of the 64 GB. **The optimizer's bookkeeping, not the model, dominates.** This is exactly why the choice of optimizer matters and why Muon - which keeps less - helps.

## The asymmetry that saves us: running is cheap

| Task | Memory (4B) |
|---|---|
| Full training | ~64 GB |
| Running (inference) | ~8 GB (or ~2 GB compressed) |

Running a model needs roughly **8x less** memory than training it. A trained 4B model runs fine on your laptop - it's *training* that overflows. STRATUM's job: make training behave more like running - small footprint - while still teaching new skills.

## Lever 1 - train few dials (LoRA)

Freeze all 4 billion dials. Add a tiny set of new ones - perhaps 20 million, a fraction of a percent - and train only those. Now gradients and optimizer notes exist only for the 20 million, not the 4 billion. The three giant rows shrink ~99%. This is **LoRA**, and it's also what makes STRATUM's tiles possible - those tiny dial-sets are the strata you train separately and merge. Full detail in doc 3.

## Lever 2 - a lean optimizer (Muon)

The optimizer's two notes cost 32 of the 64 GB. Newer optimizers keep less. **Muon** keeps one note instead of two and, by making each step higher-quality, needs *fewer steps* - a direct time saving on a slow laptop. Full detail in doc 4.

## Bonus lever - compression (quantization)

Store the frozen weights at 4 bits per number instead of 16, cutting the weights row from ~8 GB to ~2 GB. Since frozen weights are only read, never adjusted, this costs almost no quality. LoRA + this compression is the well-known **QLoRA**, and it's what fits a 4B model on an 8 GB card.

## The bottom line

| Approach | Rough VRAM (4B) | Fits a laptop? |
|---|---|---|
| Full training, AdamW | ~64 GB | No |
| LoRA, AdamW | ~18 GB | High-end desktop |
| LoRA, Muon | ~14 GB | Desktop / big laptop |
| **QLoRA + Muon** | **~6-8 GB** | **Yes - your laptop** |

That bottom row is STRATUM. Every default in the code keeps you there.

## What you now know

- The binding constraint is **VRAM**, a hard wall.
- Full training a 4B model needs **~64 GB**, mostly **optimizer bookkeeping**.
- **Running** is ~8x cheaper than training - the asymmetry we exploit.
- Three levers get you to laptop size: **LoRA** (few dials), **Muon** (lean optimizer), **quantization** (compression).

Next: [the skill-tile idea that turns a big model into small independent pieces ->](02-the-stratum-idea.md)
