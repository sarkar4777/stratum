# 5 - Merging: fusing strata into one model

*The idea STRATUM is named for. The math of why it works, the three methods and when to use each, the code, and - honestly - where it breaks.*

---

## The picture

You've trained three strata, each on its own skill, each against the same frozen base:

- `strata/extract` - pulls structured fields from documents
- `strata/classify` - sorts tickets into categories
- `strata/policy` - answers following your compliance rules

Each is a small folder. Now you want one model doing all three. That's **merging** (fusing).

```mermaid
flowchart LR
    W["Frozen base<br/>weights W"]
    D1["delta extract"] --> P((+))
    D2["delta classify"] --> P
    D3["delta policy"] --> P
    W --> P
    P --> M["Merged model<br/>W + all deltas"]

    classDef delta fill:#7F77DD,stroke:#1b1035,color:#fff
    classDef base fill:#2d1a52,stroke:#1b1035,color:#fff
    classDef out fill:#1D9E75,stroke:#1b1035,color:#fff
    class D1,D2,D3 delta
    class W base
    class M out
```

Each stratum contributes a small delta, and merging adds them all onto the same frozen base. The whole operation is addition, which is why it's cheap and why the pieces compose.

## Why it works - the math

From doc 3, a stratum's effect is **added** to the model: `W + delta`, where `delta = scaling x (B @ A)`. Addition composes - you can add several:

```
one skill: W + delta_extract
all skills: W + delta_extract + delta_classify + delta_policy
```

That's the entire trick. Every stratum is an additive adjustment to the same `W`, so stacking them is adding them. STRATUM's demo shows the real arithmetic:

```
base weight norm: 0.746
skill-1 delta norm: 0.120
skill-2 delta norm: 0.103
 weights (1.0, 1.0) -> merged norm 0.760   <- base + both skills
```

You built a capable model in pieces, and joining was addition. That's why it fit on a laptop: you never held more than one stratum's training in memory, yet the finished model contains all of them.

## Recovering the delta from a saved stratum

STRATUM stores strata as PEFT adapters (separate `lora_A` and `lora_B` tensors). At merge time it reconstructs each delta. From `stratum/merge.py`:

```python
A = raw["...q_proj.lora_A.weight"] # [r, in]
B = raw["...q_proj.lora_B.weight"] # [out, r]
scaling = lora_alpha / r # classic LoRA scaling
delta = (B @ A) * scaling # [out, in], same shape as the base weight
```

This is verified in the test suite (`test_extract_deltas_math`) to match exactly. Getting the `scaling` right is essential - miss it and every merged skill is silently mis-weighted.

## Weighted merging

Plain addition treats skills as equally important. Usually you want control, so STRATUM uses a **weighted** sum:

```
W + 0.7-delta_extract + 0.5-delta_classify + 0.3-delta_policy
```

Turn a skill up if the merged model is weak at it, down if it's bleeding into others. Default is equal weights, override with `--weights`.

## The three methods, escalating

Weighted addition (linear) is the default and right first choice. When strata conflict, smarter recipes help. STRATUM ships all three.

**1. Linear (weighted sum).** Fast, simple. Best when skills are fairly independent - extraction and classification rarely fight. This is your default.

**2. TIES - resolve conflicts.** When two strata push the *same* dial in *opposite* directions, plain addition lets them cancel and both skills suffer. TIES keeps each stratum's most important adjustments, then for each dial lets the majority sign win instead of averaging to mush. Use when linear makes several skills mediocre at once - the signature of conflict. Tune with `--density` (fraction of each stratum kept, default 0.2). The method comes from Yadav et al., "TIES-Merging" (2023).

**3. DARE - reduce interference.** DARE randomly drops most of each stratum's small adjustments and rescales the survivors. Sounds destructive, but isn't - most tiny adjustments are redundant, and dropping them cuts crosstalk between strata. Best when fusing *many* strata. Often combined with TIES. Tune with `--drop` (fraction dropped, default 0.9). The method comes from Yu et al., "DARE" (2023). Because DARE drops entries *randomly*, STRATUM seeds that randomness (`--seed`, default 42) - the same command always produces the same model, which matters when a build must be reproducible for a client.

```bash
stratum merge strata/extract strata/classify --out models/m --method linear
stratum merge strata/extract strata/classify --out models/m --method ties --density 0.3
stratum merge strata/*/ --out models/m --method dare --drop 0.9
```

You don't memorize these - you escalate: linear, check eval. If skills degrade, TIES. If many strata clash, DARE.

## Where merging fails - the honest part

**Different bases don't merge.** A stratum is tuned to *its* base's specific dials. Add it to a different base and it's nonsense. STRATUM records each stratum's base in `stratum_card.json` and **refuses** to merge mismatched ones. All strata you fuse must share one base.

**Deeply conflicting skills won't co-exist.** One stratum trained to always be terse and one to always be verbose fuse into a muddled compromise. The fix isn't a better algorithm - it's recognizing they shouldn't be one model. Keep them as separate strata and load whichever you need.

**Too many strata dilute.** 3-5 fuse cleanly. At 20, each signal goes faint. For many skills, group related ones, or keep some as swappable separate strata rather than fusing everything.

**No emergent skills.** The merged model does the *union* of what its strata taught - nothing new appears from combination. Obvious, but worth stating so expectations are right.

**QLoRA leaves a small seam.** A stratum trained with the base compressed to 4 bits (the default on GPU) learned its delta against *slightly rounded* weights, but merging applies that delta to the *full-precision* base. The mismatch is tiny and usually costs nothing measurable - STRATUM prints a note at merge time so you know it's there. If a QLoRA-trained skill scores noticeably lower after merging than it did alone, retrain that one stratum with `--no-4bit` and merge again.

## Memory, and merging strata you didn't make

A dense delta is the full size of the weight it adjusts, so "reconstruct every delta for every stratum, then add" would need several complete model copies in memory - exactly what a laptop doesn't have. STRATUM instead keeps each stratum as its small A and B factors and materializes deltas **one weight at a time** while applying them, so peak memory is the base model plus a single layer. Merging ten strata costs barely more than merging two.

Because strata are meant to be shared - between teams, or from a library of past client work - merging also treats them as untrusted input:

- adapter weights in the old pickle format are loaded with `weights_only=True`, which cannot execute code hidden in the file
- strata trained with DoRA or with fully-retrained modules (`modules_to_save`) are **refused** with a clear error, because their changes are not plain additive deltas and merging them this way would be silently wrong

Still, a stratum is model weights: only merge strata whose provenance you trust, the same judgment you apply to any third-party artifact.

## The code path, end to end

`stratum merge` calls `merge_strata()` in `stratum/merge.py`:
1. Read each stratum's `stratum_card.json` and abort if bases differ.
2. `load_stratum_factors()` on each stratum -> the small `(A, B, scaling)` pairs.
3. Load a fresh copy of the base, then for each adapted weight: materialize that one weight's deltas, combine them with the chosen method, add the result onto the base weight, move on.
4. Save a standalone model plus a `stratum_merge.json` record of exactly what was fused, with which method, weights, and seed.

The whole thing is verified in `tests/test_pipeline.py` against a tiny self-built model: train two strata, extract, merge with all three methods, check the merged weights equal base plus deltas to numerical precision, and generate. It runs on CPU in seconds - read it to see the pipeline as code.

## What you now know

- Merging works because strata are **additive** deltas on the same base: combine by adding.
- The delta is `(B @ A) x (alpha/r)` - getting the **scaling** right is essential.
- **Weighted** merging emphasizes skills, and three methods escalate: **linear -> TIES -> DARE**.
- It **fails** on different bases, deeply conflicting skills, or too many strata.

Next: [the training internals - batching, the loss mask, gradient accumulation, and every default explained ->](06-training.md)
