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

STRATUM does this in `stratum/data.py`. The boundary between prompt and response must be *exact*, so the two parts are tokenized separately and joined - never tokenized as one string and then guessed apart, because tokenizers sometimes merge characters across the seam:

```python
prompt_ids = tokenizer(prompt_text)["input_ids"]     # rendered chat prompt
response_ids = tokenizer(response_text)["input_ids"] # response + end marker
input_ids = prompt_ids + response_ids
labels = [-100] * len(prompt_ids) + response_ids     # -100 = "ignore in loss"
```

**If you get this wrong**, the model trains to generate prompts and questions instead of answers. There's no error message. Loss looks normal. The model is just quietly worse. STRATUM's test `test_loss_mask_covers_prompt_only` guards this, and you can see it yourself: a healthy batch has 40-80% of tokens masked.

Two related guards live in the same code. If a prompt alone is longer than `--max-len`, that row would be *entirely* masked - the model would "train" on it and learn nothing - so STRATUM refuses with a clear error instead of silently skipping. And if a row's response gets cut by `--max-len`, whatever survives still trains normally.

## A wrinkle: thinking models

Some recent models (Qwen3 among them, including STRATUM's default base) are **thinking models**: before answering they write out reasoning inside `<think>...</think>` tags. Useful for hard open-ended questions, wrong for a specialized extractor that should answer `{"total": 88}` and stop - the thinking burns tokens and time, and your training data has no thinking in it.

STRATUM handles this for you everywhere: chat templates are rendered with thinking disabled during training AND inference so the two match, and any think block a model still emits is stripped before scoring, chatting, or writing teacher-generated training data. You don't have to do anything - but now you know why the code passes `enable_thinking=False` around.

## The training loop, annotated

From `stratum/train.py`, the heart of it:

```python
for epoch in range(epochs):
    random.shuffle(rows)                 # different order each epoch
    for step, (input_ids, attn, labels) in enumerate(batches):
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        if not torch.isfinite(out.loss): # safety: skip bad batch
            continue
        loss = out.loss / grad_accum     # scale for accumulation
        loss.backward()                  # compute gradients
        if (step + 1) % grad_accum == 0: # every grad_accum batches:
            clip_grad_norm_(params, 1.0) # clip to prevent explosions
            for opt in opts:
                opt.step()               # update the dials
                opt.zero_grad()          # reset for the next accumulation
```

Four things in there matter and are explained next.

## Batching

Processing one example at a time wastes the GPU. A **batch** processes several at once. STRATUM pads examples in a batch to equal length and builds an attention mask so padding is ignored. `--batch-size` controls how many - bigger is faster but uses more memory.

## Gradient accumulation

You want a large *effective* batch for stable training but can't fit one in memory. So process a small batch, keep its gradients, process another, add to the stored gradients, and only update after several. The effective batch is:

```
effective_batch = batch_size x grad_accum
```

STRATUM defaults to `batch_size=4`, `grad_accum=4` -> effective 16. On a tight GPU, drop `batch_size` to 1 and raise `grad_accum` to 16 - same effective batch, less memory.

## Gradient checkpointing

During the backward pass the model normally keeps every intermediate result from the forward pass, which costs a lot of memory on long sequences. **Gradient checkpointing** throws most of them away and recomputes them when needed - a little extra compute for a big activation-memory saving. STRATUM turns it on automatically. It's the reason the activation term in `stratum plan`'s estimate stays small.

## Gradient clipping

Occasionally a batch produces enormous gradients that would lurch the weights into nonsense. **Clipping** caps the gradient size before the update. STRATUM clips to norm 1.0 every step. This, plus Muon's NaN guard (doc 4), is why STRATUM training is stable even on tiny or messy datasets.

## The system prompt travels with the model

If you train with `--system "You are a precise assistant..."`, that instruction is part of what the model learned to expect. Use the **same** system prompt at eval, chat, and serving time - the recipe's `system` key applies it consistently across training and eval gates for exactly this reason. Changing it in production quietly shifts behavior away from what you measured.

## Every default, explained

| Setting | Default | Why |
|---|---|---|
| `--base` | Qwen3-1.7B | Small enough for most laptops, run `stratum doctor` for yours |
| `--rank` | 16 | Right for style/format skills, raise to 32-64 for knowledge |
| `--lr` (Muon) | 2e-2 | Muon tolerates larger LRs than AdamW |
| `--adamw-lr` | 1e-3 | For the params Muon skips, or everything with `--optimizer adamw` |
| `--epochs` | 3 | Enough passes for a small skill dataset without over-memorizing |
| `--batch-size` | 4 | Balance of speed and memory |
| `--grad-accum` | 4 | Effective batch 16 |
| `--max-len` | 1024 | Covers most prompt+response pairs, raise for long documents |
| `--optimizer` | muon | Lighter and fewer steps, `adamw` for the conservative choice |
| `--seed` | 42 | Same seed = same data order, init, and dropout. GPU math can still differ by a hair between runs and machines - identical *quality*, not always identical bits |
| 4-bit | on (GPU) | QLoRA, to fit bigger bases, `--no-4bit` to disable |

## Checkpoints: a killed run keeps its last epoch

The adapter is saved after every epoch, not just at the end - it's only a few megabytes, and it means a run that dies (power, memory, or you stopping it) leaves the last completed epoch on disk as a usable stratum. The `stratum_card.json` records `epochs_completed` alongside the planned `epochs`, so you can always tell a partial artifact from a finished one.

## Reading a healthy run

```
trainable params: 4,505,600 || all params: 1,720,000,000 || trainable%: 0.26
Optimizer: Muon on 98 matrices, AdamW on 0 other params.
epoch 1/3 avg loss 1.4210
epoch 2/3 avg loss 0.7982
epoch 3/3 avg loss 0.5113
```

- `trainable% 0.26` - you're training a quarter of one percent. That's LoRA working.
- Loss falling smoothly each epoch - healthy. Flat loss means LR too low or rank too low for the skill. Rising eval loss (doc 8) means over-training - lower epochs.

## Watch for over-specialization

Training hard on one narrow skill can make the model worse at everything else. Guard: keep a few general questions and check them after training (doc 8). If general ability collapsed, lower epochs or rank.

## What you now know

- The **loss mask** trains only on the response - getting it wrong silently ruins fine-tunes.
- **Batching** + **gradient accumulation** give a stable effective batch within your memory.
- **Gradient clipping** and Muon's NaN guard keep training stable.
- Every default has a reason, and you now know how to change each for your case.

Next: [distillation - teaching a small model from a big one ->](07-distillation.md)
