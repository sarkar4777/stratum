# 6 - Training internals

*What actually happens when you run `stratum train`, every setting explained, and the one correctness detail that silently ruins fine-tunes if you get it wrong.*

---

## The one correctness detail: the loss mask

Your training pair is a prompt and a response. It becomes one flat string the model reads:

```
User: extract the total from 'Total: $88'
Assistant: {"total": 88}
```

**Which part should the model be trained to produce?** Only the response. The prompt is *context* - the model should learn to respond to it, not to generate it. You enforce this with a **loss mask**: mark the prompt tokens as "ignore" (the value `-100`) so loss is computed only on the response.

STRATUM does this in `stratum/data.py`. It tokenizes the prompt-only text to find where the response starts, then masks everything before it:

```python
prompt_ids = tokenizer(prompt_text)["input_ids"]
full_ids = tokenizer(full_text)["input_ids"]
labels = list(full_ids)
for i in range(len(prompt_ids)): # mask the prompt portion
    labels[i] = -100 # -100 = "ignore in loss"
```

**If you get this wrong**, the model trains to generate prompts and questions instead of answers. There's no error message. Loss looks normal. The model is just quietly worse. STRATUM's test `test_loss_mask_covers_prompt_only` guards this, and you can see it yourself: a healthy batch has 40-80% of tokens masked.

## The training loop, annotated

From `stratum/train.py`, the heart of it:

```python
for epoch in range(epochs):
    random.shuffle(rows) # different order each epoch
    for step, (input_ids, attn, labels) in enumerate(batches):
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        if not torch.isfinite(out.loss): # safety: skip bad batch
            continue
        loss = out.loss / grad_accum # scale for accumulation
        loss.backward() # compute gradients
        if (step + 1) % grad_accum == 0: # every grad_accum batches:
            clip_grad_norm_(params, 1.0) # clip to prevent explosions
            for opt in opts: opt.step(); opt.zero_grad() # update, reset
```

Four things in there matter and are explained next.

## Batching

Processing one example at a time wastes the GPU. A **batch** processes several at once. STRATUM pads examples in a batch to equal length and builds an attention mask so padding is ignored. `--batch-size` controls how many; bigger is faster but uses more memory.

## Gradient accumulation

You want a large *effective* batch for stable training but can't fit one in memory. So process a small batch, keep its gradients, process another, add to the stored gradients, and only update after several. The effective batch is:

```
effective_batch = batch_size x grad_accum
```

STRATUM defaults to `batch_size=4`, `grad_accum=4` -> effective 16. On a tight GPU, drop `batch_size` to 1 and raise `grad_accum` to 16 - same effective batch, less memory.

## Gradient clipping

Occasionally a batch produces enormous gradients that would lurch the weights into nonsense. **Clipping** caps the gradient size before the update. STRATUM clips to norm 1.0 every step. This, plus Muon's NaN guard (doc 4), is why STRATUM training is stable even on tiny or messy datasets.

## Every default, explained

| Setting | Default | Why |
|---|---|---|
| `--base` | Qwen3-1.7B | Small enough for most laptops; run `stratum doctor` for yours |
| `--rank` | 16 | Right for style/format skills; raise to 32-64 for knowledge |
| `--lr` (Muon) | 2e-2 | Muon tolerates larger LRs than AdamW |
| `--epochs` | 3 | Enough passes for a small skill dataset without over-memorizing |
| `--batch-size` | 4 | Balance of speed and memory |
| `--grad-accum` | 4 | Effective batch 16 |
| `--max-len` | 1024 | Covers most prompt+response pairs; raise for long documents |
| `--optimizer` | muon | Lighter and fewer steps; `adamw` for the conservative choice |
| `--seed` | 42 | Reproducibility - same seed, same result |
| 4-bit | on (GPU) | QLoRA, to fit bigger bases; `--no-4bit` to disable |

## Reading a healthy run

```
trainable params: 4,505,600 || all params: 1,720,000,000 || trainable%: 0.26
Optimizer: Muon on 98 matrices, AdamW on 0 other params.
epoch 1/3 avg loss 1.4210
epoch 2/3 avg loss 0.7982
epoch 3/3 avg loss 0.5113
```

- `trainable% 0.26` - you're training a quarter of one percent. That's LoRA working.
- Loss falling smoothly each epoch - healthy. Flat loss means LR too low or rank too low for the skill. Rising eval loss (doc 7) means over-training - lower epochs.

## Watch for over-specialization

Training hard on one narrow skill can make the model worse at everything else. Guard: keep a few general questions and check them after training (doc 7). If general ability collapsed, lower epochs or rank.

## What you now know

- The **loss mask** trains only on the response; getting it wrong silently ruins fine-tunes.
- **Batching** + **gradient accumulation** give a stable effective batch within your memory.
- **Gradient clipping** and Muon's NaN guard keep training stable.
- Every default has a reason, and you now know how to change each for your case.

Next: [distillation - teaching a small model from a big one ->](07-distillation.md)
