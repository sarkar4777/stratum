# 7 - Distillation: teaching a small model from a big one

*One of the most powerful ways to build a good small model. Explained from zero for a developer who has never touched machine learning, then shown as real STRATUM commands and code.*

---

## The idea, by analogy

Imagine a master craftsperson (the **teacher**) and an apprentice (the **student**). The apprentice could learn purely from a rulebook (fixed correct answers). But they learn far faster by watching the master work - seeing not just *what* the master does, but *how confident* the master is, which alternatives they considered, where they hesitated.

**Distillation** is exactly this for models. You have a big, capable **teacher** model and a small **student** model. Instead of training the student only on fixed human-written answers, you train it to imitate the teacher. The student ends up much better than it would from the rulebook alone - often nearly as good as the teacher on the specific skill, at a fraction of the size and cost.

This is one of the main ways strong small models are actually built in industry. If you've heard "this 3B model punches above its weight," distillation from a bigger model is often why.

## Why it works: the teacher's "soft" knowledge

Here's the insight that makes distillation more than just "copy the teacher's answers."

When a model predicts the next token (doc 0), it doesn't just pick one - it produces a *probability for every possible token*. Ask a model to classify a ticket and it might output:

```
billing 88%
account_access 9%
bug 2%
how_to 1%
```

The single correct answer is "billing." A rulebook teaches only that. But the *full distribution* carries much richer information: it says billing is likely, account_access is a plausible near-miss, and bug/how_to are basically wrong. That shape - the teacher's **uncertainty** - is knowledge. It tells the student *how* to think about the problem, not just the final answer.

Training the student to match the teacher's whole distribution (called the **soft labels**) instead of just the one correct token (the **hard label**) is what gives distillation its power. The student learns the teacher's judgment, not just its conclusions.

## Two ways to distill (STRATUM supports both)

There are two flavors, and picking the right one matters. STRATUM gives you both.

```mermaid
flowchart TB
    subgraph Data["Data distillation (simple)"]
        direction TB
        SI["Seed inputs"] --> TE["Teacher<br/>(any model, even an API)"]
        TE --> TP["Training pairs"]
        TP --> ST1["Train a normal stratum"]
    end

    subgraph Logit["Logit distillation (advanced)"]
        direction TB
        TX["Same text"] --> TM["Teacher model"]
        TX --> SM["Student model"]
        TM -->|"soft probabilities"| KL{{"Match distributions<br/>(KL divergence)"}}
        SM --> KL
        KL --> ST2["Trained student stratum"]
    end

    classDef teacher fill:#D85A30,stroke:#1b1035,color:#fff
    classDef student fill:#1D9E75,stroke:#1b1035,color:#fff
    class TE,TM teacher
    class SM,ST1,ST2 student
```

Data distillation (top) has the teacher write your training data, then trains a normal stratum - the teacher can be anything, including a closed API. Logit distillation (bottom) runs teacher and student together and trains the student to match the teacher's full probability distribution - richer, but they must share a tokenizer.

### Flavor 1 - Data distillation (simple, recommended to start)

The teacher **writes your training data**. You give it a pile of example inputs (documents, tickets, questions), the teacher produces an ideal answer for each, and those input->answer pairs become a normal training set you feed to `stratum train`.

- **Pro:** dead simple, robust, and the teacher can be *anything* - including a closed API like GPT or Claude that you can't run locally. You never need teacher and student to be compatible.
- **Con:** the student learns only the teacher's final answers (the hard labels), not its full uncertainty. Still very effective - this gets you most of the benefit.

This is the right default for almost everyone.

### Flavor 2 - Logit distillation (advanced, maximum quality)

Teacher and student run on the same text *at the same time*, and the student is trained to match the teacher's full probability distribution (the soft labels) directly. The name: a **logit** is the raw score a model assigns to each vocabulary token before those scores become probabilities - this flavor matches the teacher at that raw-score level.

- **Pro:** the richest possible signal - the student learns the teacher's uncertainty. Best final quality.
- **Con:** teacher and student must **share a tokenizer** (be from the same model family, e.g. both Qwen3), and *both* must fit in memory together. More setup, more hardware.

Use this when you have a capable local teacher from the same family as your student and want to squeeze out the last bit of quality.

## Data distillation - the commands

Say you want an invoice-extraction skill and you have a big teacher to generate clean examples.

**Step 1 - collect seed inputs.** A text file, one raw input per line (`seeds.txt`):

```
Subtotal $80, Tax $8, Total $88
Amount due: $250.00
Grand total: 1,499 EUR
```

**Step 2 - have the teacher write the training pairs:**

```bash
# Using a local Hugging Face teacher model:
stratum teacher-gen \
  --seeds seeds.txt \
  --instruction "Extract the invoice total as JSON like {\"total\": N}." \
  --teacher hf --model Qwen/Qwen3-4B \
  --out examples/extract_distilled.jsonl

# Or using an API teacher (set the key first):
export OPENAI_API_KEY=sk-...
stratum teacher-gen --seeds seeds.txt \
  --instruction "Extract the invoice total as JSON." \
  --teacher openai --model gpt-4o-mini \
  --out examples/extract_distilled.jsonl
```

Provider model names age quickly - check your provider's current model list and pass `--model` explicitly rather than trusting an example or a built-in default to stay current.

STRATUM asks the teacher for each seed and writes a `{"prompt","response"}` JSONL. The four teacher backends are `hf` (local model), `openai`, `anthropic`, and `echo` (a no-op for testing the pipeline).

Three details matter when you scale this to thousands of seeds:

- **Each pair is written the moment it exists.** A crash or network drop at seed 4,999 of 5,000 loses one pair, not the run.
- **Failed calls retry with growing pauses**, and re-running the same command **resumes** - seeds already answered in the output file are skipped. Generating a big dataset against a flaky API is a matter of re-running until it's done.
- **The teacher's answers arrive clean.** If the teacher is a thinking model (doc 6), its `<think>` reasoning is stripped so your training data contains answers, not deliberation.

**Step 3 - train a normal stratum on the distilled data:**

```bash
stratum train --skill examples/extract_distilled.jsonl --out strata/extract
```

That's it. The teacher's expertise is now baked into a small, mergeable stratum. It fuses with your other strata like any other (doc 5).

## Logit distillation - the command

When you have a same-family teacher and want maximum quality:

```bash
stratum distill \
  --skill examples/extract.jsonl \
  --out strata/extract \
  --student Qwen/Qwen3-1.7B \
  --teacher Qwen/Qwen3-4B \
  --temperature 2.0 \
  --alpha 0.5
```

- `--student` learns, `--teacher` is imitated (frozen).
- `--temperature 2.0` **softens** both distributions so the student learns from the teacher's smaller, informative probabilities, not just its top pick. Higher temperature = softer = more attention to the teacher's "second thoughts." 2.0 is a good default.
- `--alpha 0.5` balances the two losses: half from matching the teacher's soft distribution, half from the true answer. 0.5 is a sane default, raise it to trust the teacher more.
- `--teacher-4bit` compresses the frozen teacher to 4 bits on an NVIDIA GPU. The teacher is only read, never trained, so this is nearly free (doc 1's quantization lever again) and it roughly quarters the teacher's memory - often the difference between "doesn't fit" and "fits".
- `--grad-accum 4` gives logit distillation the same effective-batch trick as normal training (doc 6), since its per-step batches must be small enough for two models at once.

STRATUM checks that teacher and student share a vocabulary and gives a clear error if they don't - a common beginner trap.

## What the code actually does

The heart of logit distillation is the loss function (`stratum/distill.py`). In plain terms it computes two things and blends them:

```python
# Soft loss: student should match the teacher's SOFTENED distribution.
# KL divergence measures how different two probability distributions are.
soft = KL( softmax(teacher / T) , softmax(student / T) ) * T*T

# Hard loss: ordinary next-token loss against the real answer.
hard = cross_entropy(student, true_tokens)

loss = alpha * soft + (1 - alpha) * hard
```

- **KL divergence** is just "how far apart are these two probability distributions." Minimizing it pulls the student's distribution toward the teacher's.
- The `T*T` term restores the gradient strength that temperature-softening would otherwise shrink - a standard detail, handled for you.
- Loss is computed only on the response tokens (the loss mask from doc 6 applies here too).

You don't need to memorize this - but now you can read it and explain it. That's the KL-divergence-and-temperature machinery from the original distillation paper - Hinton, Vinyals and Dean, "Distilling the Knowledge in a Neural Network" (2015) - and it's twelve lines.

## When to use distillation

- **You have access to a much better model** (an API or a big local one) and want a small deployable model that captures its skill. -> distillation, ideally data distillation first.
- **Your hand-written data is thin.** A teacher can generate thousands of consistent examples cheaply. -> data distillation.
- **You need the absolute best small-model quality and have a same-family teacher.** -> logit distillation.
- **You already have plenty of good real data and no better teacher.** -> skip distillation - plain `stratum train` is fine.

## The honest caveats

- **The student can't exceed the teacher** on the distilled skill - it's imitating. If the teacher is wrong, the student learns the mistake. Use a teacher genuinely better than your student.
- **An API teacher sees your seed inputs.** Every seed you feed `teacher-gen` with `--teacher openai` or `--teacher anthropic` is sent to that provider. If the seeds are client documents, tickets, or anything under a data-residency requirement, that transfer may itself be a compliance violation - use a **local** teacher (`--teacher hf`) so nothing leaves your environment, which is the whole promise of doc 10's production loop.
- **Data distillation inherits the teacher's licensing.** If you distill from a commercial API, check that its terms permit training a model on its outputs. This is a real legal consideration for industry work, not just a formality.
- **Logit distillation needs a shared tokenizer.** Qwen-teacher to Llama-student won't work for logit distillation (their vocabularies differ) - use data distillation across families instead.

## What you now know

- **Distillation** teaches a small **student** to imitate a big **teacher**, learning the teacher's judgment, not just its answers.
- The teacher's **soft labels** (full probability distribution) carry richer knowledge than a single correct answer.
- **Data distillation** (teacher writes the data) is simple and works across any models - start here.
- **Logit distillation** (student matches the teacher's distribution via KL divergence, softened by temperature) gives top quality but needs a same-family teacher.
- Both produce ordinary strata that **fuse like any other**.

Next: [evaluation - proving your model works instead of guessing ->](08-evaluation.md)
