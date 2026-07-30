# Example data

These files exist to teach the shapes and prove the pipeline runs, not to
produce a good model. They are deliberately tiny - eight pairs per skill where
a real skill wants hundreds (docs 9 and 10 give the numbers). Expect a model
trained on them to be unimpressive. That's fine: swap in your own data of the
same shape and everything else stays identical.

| File | What it is |
|---|---|
| `extract.jsonl` | Training pairs for an invoice-total extraction skill: `{"prompt", "response"}` per line |
| `classify.jsonl` | Training pairs for a support-ticket classification skill |
| `test-extract.jsonl` | Held-out test set for extraction: `{"prompt", "expected"}`, scored with `json_field` |
| `test-classify.jsonl` | Held-out test set for classification, scored with `exact` |
| `test.jsonl` | A small mixed set with `"skill"` labels, for a quick overall check with `contains` |
| `seeds.txt` | Raw inputs for `stratum teacher-gen`, one per line |
| `recipe.yaml` | A complete build: two strata, a merge, and eval gates |
| `recipe-distill.yaml` | The same build with one stratum distilled from a bigger teacher |

Building your own skill data:

- One JSONL file per skill, one `{"prompt", "response"}` object per line, UTF-8.
- Keep responses format-consistent - if you want JSON out, every response is valid JSON.
- Before training, set aside 10-20% of the pairs as a test file with `expected`
  instead of `response`. Never train on those (doc 8 explains why this matters).
- 100-500 pairs proves a skill, 1,000-5,000 is solid production quality (doc 10).
