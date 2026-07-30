# 13 - Troubleshooting

*The problems people actually hit, in the order they usually hit them. Every error STRATUM raises on purpose is listed here with the fix.*

---

## Out of memory during training

First, `stratum plan recipe.yaml` - it estimates what each stratum needs on this machine and suggests the fix, or tells you honestly that the build belongs on rented hardware (doc 10). The levers, in the order to pull them:

1. `--batch-size 1 --grad-accum 16` - same effective batch (doc 6), a fraction of the memory.
2. Make sure 4-bit is on. It's the default on NVIDIA GPUs, but only if `bitsandbytes` is installed - `stratum doctor` tells you.
3. `--max-len 512` if your pairs are short. Memory grows with sequence length.
4. A smaller base. Run `stratum doctor` and use its recommendation. A pipeline proven on 0.6B scales to 4B later by changing one argument.

## `bitsandbytes` won't install or won't load

4-bit (QLoRA) needs `bitsandbytes`, which needs an NVIDIA GPU. On Windows, recent versions install normally with `pip install bitsandbytes` - if an old pinned version fails, upgrade pip and retry. On Mac or CPU-only machines it isn't available: train in bf16 (`--no-4bit` makes this explicit) with a smaller base.

## Model downloads fail

Run `stratum doctor` first - it checks the exact things that go wrong:

- **Misspelled id.** Hub ids look like `Qwen/Qwen3-1.7B`, namespace and all.
- **Gated model.** Some releases need a (free) license acceptance: `huggingface-cli login`.
- **Locked-down network.** Set `HF_HUB_OFFLINE=1` and point `--base` at a local folder, after downloading on a connected machine with `huggingface-cli download <model>` and copying the folder over.

## "A training row has no response tokens within --max-len"

One of your prompts alone is longer than `--max-len`, so the response would be cut off entirely and the row would teach nothing. STRATUM stops instead of training on it silently. Raise `--max-len`, or shorten or drop that row - the error prints the start of the offending prompt.

## The model writes `<think>` blocks or reasons before answering

It's a thinking model (doc 6). STRATUM disables thinking in every template it renders and strips think blocks from output it scores or displays, so you'd normally never see one. If you're serving the merged model with your own stack (vLLM and the like), render prompts with `enable_thinking=False` there too - the serving stack doesn't know what STRATUM knows.

## A skill scores worse after merging than alone

Some merge cost is normal (a couple of points - doc 8 shows a healthy example). A collapse means conflict:

1. Try `--method ties`, then `--method dare` (doc 5 explains when each helps).
2. Turn that skill's weight up with `--weights`.
3. If two skills genuinely fight, keep them as separate swappable strata (doc 10, pattern B).
4. If the stratum was trained with 4-bit and the drop is small but real, retrain it with `--no-4bit` - see the QLoRA seam note in doc 5.

## "Strata have different base models"

Merging refuses because the strata weren't trained from the same base, and deltas against one base are nonsense against another (doc 5). Retrain the odd one out on the shared base - the error names each stratum's base so you can see which it is.

## Merging refuses a DoRA stratum or one with modules_to_save

Both change the model in ways that aren't plain additive deltas, so merging them with this math would be silently wrong - STRATUM says so instead. Retrain the stratum as a standard LoRA adapter, or keep it as a separate runtime adapter.

## `teacher-gen` died halfway through a big run

Nothing is lost. Pairs are written as they're generated, so re-run the exact same command - seeds already answered are skipped and only the missing ones go to the teacher. Failed seeds retry with growing pauses automatically.

## CPU-only training is very slow

It works, but set expectations: a few hundred pairs on a 0.6B base is an hours-not-minutes job on CPU. Use it to prove your data and pipeline, then do the real training burst on any machine with a GPU - the commands are identical.

## Training loss is flat

Either the learning rate is too low for the optimizer you chose (Muon's default is `2e-2`, AdamW wants around `1e-4` to `1e-3`), or the rank is below what the skill needs and you've hit the ceiling from doc 3. Raise `--rank` before blaming the method.

## Corpus ingest says a format needs an extra library

The document parsers are optional dependencies so the core install stays light. `pip install 'stratum-slm[corpus]'` brings PDF, Word, PowerPoint, Excel, and image support in one go.

## "no extractable text - this is probably a scanned PDF"

The PDF is pictures of pages, with no text layer to read. Two routes: run OCR over it before ingesting, or export its pages as images and ingest those with a vision teacher (`--images hf`). Doc 14 covers both.

## Something else

Open an issue with your `stratum doctor` output and the full error. Hardware reports (GPU, VRAM, which base worked) are welcome even when nothing is broken - they build the community table in CONTRIBUTING.md.
