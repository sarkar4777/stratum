# Reference build: an energy-sector SLM from a public web corpus

A complete worked example - every command that turned eleven public documents
into a specialized model, and the numbers it produced. Everything here ran on
one laptop with an 8 GB NVIDIA GPU.

The two files are the whole spec: `sources.txt` says what the corpus is,
`recipe.yaml` says what gets built from it.

## The commands

```bash
# 0. Pull the corpus down (10 Wikipedia articles + a US EIA outlook PDF)
stratum corpus fetch --urls-file examples/energy/sources.txt --out raw/

# 1. Extract, deduplicate, redact, and chunk it
stratum corpus ingest --in raw/ --out corpus/ --redact

# 2. Have teachers write the training pairs, one run per skill.
#    A local model keeps everything on your machine:
stratum corpus pairs --chunks corpus/chunks.jsonl \
  --instruction "Write questions an energy-sector engineer or analyst would ask, with precise factual answers." \
  --teacher hf --model Qwen/Qwen3-0.6B --per-chunk 2 --max-chunks 22 \
  --out data/energy-qa.jsonl --test-out data/energy-qa-test.jsonl

#    Claude through the CLI, on an existing subscription, no API key:
stratum corpus pairs --chunks corpus/chunks.jsonl \
  --instruction "Write extraction questions whose answer is a specific value, name, or short phrase found in the passage. Each answer must be a complete short sentence stating the value." \
  --teacher claude-cli --per-chunk 3 --max-chunks 20 \
  --out data/energy-extract.jsonl --test-out data/energy-extract-test.jsonl

stratum corpus pairs --chunks corpus/chunks.jsonl \
  --instruction "Write questions that ask how or why an energy-sector process or system works, with clear two-to-four sentence explanations." \
  --teacher claude-cli --per-chunk 2 --max-chunks 8 \
  --out data/energy-explain.jsonl --test-out data/energy-explain-test.jsonl

# 3. Check the build fits this machine, then run it
stratum plan examples/energy/recipe.yaml
stratum stack examples/energy/recipe.yaml

# 4. Measure against the untrained base
stratum eval models/energy-slm --test data/energy-qa-test.jsonl \
  --scorer overlap --baseline Qwen/Qwen3-1.7B --json-out reports/qa.json
```

## What it produced

Corpus: 11 documents, 304 chunks, no extraction errors.

| Skill | Base Qwen3-1.7B | Trained SLM | Gain |
|---|---|---|---|
| Domain Q&A | 8.1% | 41.2% | +33.1 |
| Extraction | 23.6% | 56.4% | +32.8 |

Scored with `overlap` (word-overlap F1), which is the scorer free-text answers
need - `contains` marks a correct paraphrase as zero. Both recipe eval gates
pass, and the merge moves the base weights 4.6% on average, comfortably inside
the healthy range doc 5 describes.

**Treat these numbers as proof that the pipeline works, not as product
quality.** The run sampled about 20 chunks per skill and the test sets are 4-12
cases. Doc 10 asks for 1,000-5,000 pairs per skill for production, and the 304
ingested chunks are there whenever you want the fuller run - raise or drop
`--max-chunks`.

## Two things this build demonstrated the hard way

**Teacher quality is the whole ballgame.** The extraction skill was first
taught by the local 0.6B model and scored **5.3%**, failing its eval gate. The
gate refused to certify the build - correctly. Regenerating that one skill's
pairs with Claude as the teacher, changing nothing else, took it to **56.4%**.
This is what doc 14 means when it says expert review of generated pairs is not
optional.

**Three full-strength strata must be averaged, not summed.** The first merge
used the default weights of 1.0 each and produced a model that generated
nothing at all - empty output on every prompt, with perfectly healthy training
losses. `normalize: true` in the recipe fixes it, and every merge now reports
how far it moved the base weights so the failure announces itself. Doc 5 has
the full story.

## Notes on the sources

Wikipedia content is CC BY-SA - keep the attribution if you redistribute
anything derived from it. The US EIA outlook is a public-domain US government
publication. Both are used here because they are freely available and safe to
share, which is exactly what a reference build needs. Your own corpus will be
your organization's documents, and the local `hf` teacher exists so that those
never leave your environment.
