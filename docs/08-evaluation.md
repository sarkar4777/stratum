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

## Running it, and choosing a scorer

```bash
stratum eval models/my-slm --test test.jsonl --scorer contains
```

STRATUM ships three scorers because "correct" means different things per task:

| Scorer | Rule | Use for |
|---|---|---|
| `contains` | expected string appears in output | lenient default, quick checks |
| `exact` | output equals expected (normalized) | classification, where extra words are wrong |
| `json_field` | parse both as JSON, score field by field | extraction, so `{"total":88,"tax":8}` scores per field |

The principle: your scorer should mark something correct only if it'd be correct in real use. If sloppy answers score well, your number lies. The scorers live in `stratum/evaluate.py` and are short - edit or add your own (numeric tolerance, regex, etc.) for your task.

## The comparison that proves your work

Run eval at three points and keep the numbers:

```
base model: extract 20% classify 35%
extract stratum alone: extract 85% classify 35% (extract learned; classify untouched)
classify stratum alone:extract 20% classify 88% (classify learned)
merged model: extract 82% classify 85% (both, slight merge cost)
```

The small 85->82 drop on extract when merged is normal - the cost of one model sharing two skills. If extract instead *collapsed* to 40%, the strata conflict: escalate the merge method (doc 5) or keep them separate.

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
- Match the **scorer** to your real task or the number misleads.
- Compare **base -> single stratum -> merged** to prove each step.
- Guard against **over-specialization** with a small general-ability check.

Next: [the full walkthrough - empty folder to working model, every command ->](09-full-walkthrough.md)
