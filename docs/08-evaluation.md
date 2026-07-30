# 8 - Evaluation

*Proving your model works instead of trusting a lucky demo. The step people skip and regret.*

---

## Why you cannot skip this

You'll be tempted to try three prompts, see good answers, and ship. That's vibes-based evaluation and it lies. A model can nail your three tries and fail the fourth. You need a **fixed test set** and a **number**, because the number answers the only questions that matter:
- Is my trained model better than the untrained one?
- Did merging make any skill worse?
- Did switching merge method help?

Without a number, each is a shrug.

## The golden rule: never test on training data

Your model has *seen* its training pairs - ask it those and it may recite, telling you nothing. Test data must be **held out**: examples never trained on. When building a skill's data, set aside 10-20% before training. Those become the test set.

## Building a test set

Same format as training, but `expected` instead of `response`:

```json
{"prompt": "Extract the total from: 'Total: $312'", "expected": "{\"total\": 312}"}
{"prompt": "Classify: 'I was charged twice'", "expected": "billing"}
```

30-100 cases per skill. Include hard ones deliberately - ambiguous inputs, edge cases, things you expect to fail. An all-easy test set flatters the model and teaches nothing.

## One test set per skill

Score each skill with the scorer that matches it, not one scorer stretched across everything. A `json_field` scorer applied to a classification answer like `billing` scores it zero every time - not because the model is wrong but because the scorer is. So keep **one test file per skill**, plus optionally a small mixed file for a quick overall look:

```bash
stratum eval models/my-slm --test examples/test-extract.jsonl --scorer json_field
stratum eval models/my-slm --test examples/test-classify.jsonl --scorer exact
```

For the mixed file, give each row a `"skill"` label and use the lenient `contains` scorer - the report then breaks the score down per skill, so a merge that hurt one skill shows up immediately:

```
Score (contains): 80.0% over 5 cases
  classify: 100.0%
  extract: 50.0%
```

## Choosing a scorer

STRATUM ships three scorers because "correct" means different things per task:

| Scorer | Rule | Use for |
|---|---|---|
| `contains` | expected string appears in output | lenient default, quick checks, mixed sets |
| `exact` | output equals expected (normalized) | classification, where extra words are wrong |
| `json_field` | parse both as JSON, score field by field | extraction, so `{"total":88,"tax":8}` scores per field |

`json_field` compares numbers as numbers: a model answering `"1,499"` against an expected `1499` is right in substance and scores as right. And for thinking models (doc 6), any `<think>` block is stripped before scoring, so you measure the answer, not the reasoning preamble.

The principle: your scorer should mark something correct only if it'd be correct in real use. If sloppy answers score well, your number lies. The scorers live in `stratum/evaluate.py` and are short - edit or add your own (numeric tolerance, regex, etc.) for your task.

## The comparison that proves your work

Run eval at three points and keep the numbers:

```
base model: extract 20% classify 35%
extract stratum alone: extract 85% classify 35% (extract learned, classify untouched)
classify stratum alone:extract 20% classify 88% (classify learned)
merged model: extract 82% classify 85% (both, slight merge cost)
```

Two things make this comparison one command each. `stratum eval` accepts a **stratum directory directly** - it notices the folder is an adapter, loads the base recorded in the stratum's card, and attaches it, so "extract stratum alone" is just `stratum eval strata/extract --test ...`. And `--baseline` runs the same test against a second model in the same command:

```bash
stratum eval models/my-slm --test examples/test-extract.jsonl \
  --scorer json_field --baseline Qwen/Qwen3-1.7B
```

```
Baseline score: 20.0%  ->  your model: 82.0% (+62.0%)
```

The small 85->82 drop on extract when merged is normal - the cost of one model sharing two skills. If extract instead *collapsed* to 40%, the strata conflict: escalate the merge method (doc 5) or keep them separate.

## Gating a build in CI

An evaluation you run by hand gets skipped on a busy day. Two flags turn it into a machine-checkable gate:

```bash
stratum eval models/my-slm --test examples/test-extract.jsonl \
  --scorer json_field --json-out reports/extract.json --min-score 0.8
```

`--json-out` writes the full report (mean, per-skill scores, every prompt and output) as JSON for dashboards or diffing. `--min-score` makes the command exit non-zero when the score falls below the bar, which is exactly what a CI pipeline needs to block a bad build. Doc 10 shows where this fits in a production loop.

## Guard against over-specialization

Keep a tiny separate set of general questions ("Capital of Japan?", "Summarize this."). Run it after training. If those answers fall apart, you over-trained - lower epochs or rank. A narrow model that forgot how to be a model is a common, avoidable failure.

## What a real evaluation report looks like

```
OK score 1.00 expected: {"total": 44} got: {"total": 44}
~~ score 0.50 expected: {"total":88,"tax":8} got: {"total": 88}
XX score 0.00 expected: billing got: account

Score (json_field): 71.4% over 20 cases
```

`OK`/`~~`/`XX` mark full/partial/no credit - the partial line shows `json_field` catching one of two fields, which a `contains` scorer would have scored 0 and hidden.

## What you now know

- Evaluate with a **fixed, held-out test set** and a **number** - never vibes, never training data.
- Match the **scorer** to your real task or the number misleads - one test file per skill, `"skill"` labels for mixed sets.
- Compare **base -> single stratum -> merged** to prove each step, with `--baseline` and direct stratum eval doing the legwork.
- Gate builds in CI with `--json-out` and `--min-score`.
- Guard against **over-specialization** with a small general-ability check.

Next: [the full walkthrough - empty folder to working model, every command ->](09-full-walkthrough.md)
