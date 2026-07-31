# 0 - What is a language model?

*Start here. Assumes zero machine-learning background. By the end you'll understand what a model is, what it's made of, what training means, and enough vocabulary to read every later doc without gaps.*

---

## One-sentence version

A language model is a machine that, given some text, predicts what text comes next. Everything else is that prediction, repeated.

## The prediction, concretely

Type `The capital of France is` and the model produces a ranked list of guesses for the next chunk of text, each with a confidence:

```
Paris 91% | the 3% | a 2% | Lyon 1% | ...(thousands more, tiny)
```

It appends the top guess and runs again. A model writing a paragraph is this loop firing hundreds of times. That's the whole behavior - chat, code, extraction, all of it.

## Tokens: the chunks it works in

Models don't read letters or words. They read **tokens** - chunks of text averaging about 3/4 of a word. Common words are one token, rarer strings split up:

```
"cat" -> [cat]
"caterpillar"-> [cater][pillar]
"50Hertz" -> [50][H][ertz]
```

The model has a fixed **vocabulary** of tokens it knows (often ~150,000). It emits a score for *every* vocabulary token at each step - that's the "thousands more" above. Two consequences matter later:
1. Your industry jargon may split into many tokens, making prompts and outputs longer than they look - worth remembering when you set sequence lengths (doc 6) and read costs.
2. Everything the model reads or writes is a sequence of these token IDs.

## Parameters: knowledge stored as numbers

Inside the model are billions of numbers called **parameters** (collectively the **weights**). Text enters as numbers, flows through the parameters via a fixed sequence of multiplications, and out come the next-token scores.

- "A 4-billion-parameter model" (**4B**) has four billion of these numbers.
- The parameters *are* the knowledge. There's no parameter labeled "France fact" - knowledge is spread across billions of numbers no human reads directly.

These docs call parameters **dials**, so pin the picture down now: a dial like the volume knob on a speaker - something with a current setting that you can turn up or down. Each parameter is one number, its value is the dial's current position, and changing the number is turning the dial. A model is billions of these knobs, and their combined positions determine every output. **Training is nothing more than turning the dials to better positions.** Whenever a later doc says dial, read parameter - same thing, friendlier word.

## The shape inside: layers, and the word "transformer"

Those billions of dials aren't a loose heap. They're organized into **layers**, and a layer is a simple thing: a grid of dials that the numbers get multiplied through, followed by a simple squashing function that keeps values in range. Layers **stack** - the output of one becomes the input of the next, and that stack is the "fixed sequence of multiplications" from a moment ago. Early layers end up handling surface patterns (spelling, common phrases) and later layers end up handling meaning, though nobody assigns those jobs - they emerge from training.

The specific stack design every model you'll touch uses is called a **transformer**. Its layers come in pairs of steps: an **attention** step, where each token looks back at the other tokens in the input to decide which ones matter for predicting what comes next, and a plain processing step of the multiply-through-grids kind. A model like Qwen3-1.7B is a few dozen of these layers stacked, and that's the whole architecture story you need for this project - STRATUM never modifies the design, it adjusts the dials inside the grids. When doc 3 attaches small adapters "to every linear grid," those grids are the ones inside these layers.

## Training: four-step loop

This loop is the foundation of the entire project. Learn it once.

1. **Predict.** Show the model text where you know what comes next. It guesses.
2. **Measure error.** Compare guess to truth. The error as a single number is the **loss**. Low probability on the right answer means high loss.
3. **Assign blame.** For every dial, compute whether nudging it up or down would lower the loss. Each dial's answer is its **gradient**.
4. **Nudge.** Move every dial a small step in its loss-lowering direction.

Repeat millions of times and the dials settle into good values. That's all training is. The component doing step 4 - deciding *how far* to nudge each dial - is the **optimizer**. STRATUM uses one called **Muon** (doc 4).

## Where models come from - and why you start partway

The model you download already went through this loop on trillions of tokens, costing millions of dollars. That giant first pass is **pretraining**, and it gives the model language and general knowledge.

**You will not pretrain.** You start from someone's pretrained model (free to download) and do a small, cheap amount of extra training to specialize it. That's **fine-tuning**, and it's laptop-affordable - especially STRATUM's way.

## Base vs instruct models

- A **base model** is the raw pretrained result: it completes text but doesn't follow instructions.
- An **instruct model** has extra training to answer and follow instructions.

STRATUM usually starts from an instruct model, then layers your specific skills on top.

## What you now know

- A model **predicts the next token**, repeatedly - that's how it writes.
- It's billions of **parameters** (dials) holding everything it knows, organized into stacked **layers** - the stack design is called a **transformer**.
- **Training** is a four-step loop lowering **loss**, and the **optimizer** does the nudging.
- You start from a **pretrained** model and do cheap **fine-tuning**, not pretraining.

Next: [why fine-tuning on a laptop is hard, with the exact numbers ->](01-the-memory-problem.md)
