# 11 - Glossary

*Every term in STRATUM, with full form and plain definition.*

**Attention** - The step in a transformer layer where each token looks back at the other tokens to decide which ones matter for the next prediction. STRATUM never modifies it, but the weight grids inside it are among those LoRA adapts.

**Adapter** - A small trainable add-on attached to a frozen model that adjusts it for one skill. In STRATUM, a "stratum." Often under 1% of the model's size.

**AdamW (Adam with decoupled Weight decay)** - The standard optimizer. Keeps two running notes per parameter (momentum, variance) - effective but memory-heavy. STRATUM uses Muon for weight matrices and a small AdamW for the rest.

**Backward pass (backpropagation)** - The bookkeeping sweep after each prediction that computes every parameter's gradient, walking backward through the layers. The forward pass makes the prediction, the backward pass assigns the blame.

**Base model** - The pretrained model you start from and specialize. All strata you merge must share one base.

**Batch (minibatch)** - Several training examples processed together for GPU efficiency. "Minibatch" stresses that it's a small random sample of the dataset, which is what makes gradients noisy estimates - and training "stochastic".

**bf16 (brain float 16)** - Storing each number in 2 bytes - the common training format.

**Catastrophic forgetting / over-specialization** - When training hard on a narrow skill makes the model worse at everything else. Guard by limiting epochs and checking general ability.

**Chunk / chunking** - Cutting a long document into overlapping windows a teacher model can read whole (about 2,400 characters here). The overlap keeps facts that straddle a cut intact in at least one chunk.

**Corpus** - An organization's pile of real documents and images - the raw material the corpus pipeline (doc 14) turns into training data.

**CUDA** - NVIDIA's compute platform, the way PyTorch talks to NVIDIA GPUs. "CUDA available" means PyTorch can see and use your NVIDIA card - and a CPU-only PyTorch build reports it false even with a card present (doc 13 has the fix).

**DARE** - A merge method that randomly drops most of each stratum's small adjustments and rescales the rest, reducing crosstalk when fusing many strata. From Yu et al. (2023). STRATUM seeds the randomness so the same merge always gives the same model.

**Delta** - A stratum's weight adjustment, `scaling x (B @ A)`, added onto a base weight during merging.

**Dial** - These docs' word for a parameter: one adjustable number inside the model, pictured as a volume knob with a current position. Turning a dial means changing that number. Training turns billions of them.

**Distillation** - Training a small "student" model to imitate a large "teacher" model, so the student captures the teacher's skill cheaply. STRATUM supports data distillation (teacher writes the training pairs) and logit distillation (student matches the teacher's probability distribution).

**Data distillation** - The simple flavor: a teacher model generates the training pairs, which then train a normal stratum. Works with any teacher, including closed APIs.

**Layer** - One grid of dials the numbers multiply through, plus a simple squashing function. Layers stack - each one's output feeds the next - and a model is a few dozen of them.

**Learning rate** - The master knob for how big each training step is (`--lr`). Too small and training crawls, too large and the dials overshoot and never settle. Muon's default here is 2e-2, AdamW's 1e-3 - different optimizers want very different values.

**Logit** - The raw score a model assigns to one vocabulary token before scores are turned into probabilities. "The logits" are the full row of scores, one per token in the vocabulary.

**Logit distillation** - The advanced flavor: teacher and student run together and the student is trained to match the teacher's full softened distribution (soft labels). Needs a shared tokenizer.

**Soft labels** - A teacher's full probability distribution over next tokens (e.g. billing 77%, account_access 17%, ...), which carries richer information than the single correct answer (the hard label). The signal logit distillation learns from.

**KL divergence (Kullback-Leibler divergence)** - A measure of how different two probability distributions are. Logit distillation minimizes it to pull the student's distribution toward the teacher's.

**Temperature (distillation)** - A factor that softens probability distributions so the student learns from the teacher's smaller, informative probabilities, not just its top pick. Higher = softer.

**Teacher / student** - In distillation, the large capable model being imitated (teacher) and the small model being trained (student).

**EOS token (end of sequence)** - The special token a model emits to say "I'm done answering." Training pairs end with it so the fine-tuned model learns to stop.

**Epoch** - One full pass over the training data.

**Fine-tuning** - Further training of a pretrained model to specialize it. Cheap. What STRATUM does.

**Gradient** - For each parameter, the direction that would lower the loss. Computed in the backward pass.

**Gradient accumulation** - Summing gradients over several small batches before one update, to get a large effective batch within limited memory.

**Gradient checkpointing** - Recompute some intermediate results during the backward pass instead of storing them all, trading a little compute for a lot of activation memory. On by default in STRATUM training.

**Gradient clipping** - Capping gradient size before an update to prevent instability.

**Greedy decoding** - Generating by always taking the model's top-scoring token, with no randomness. STRATUM evaluates and chats greedily so results repeat run to run.

**Held-out set** - Data set aside before training and never trained on. The only honest thing to evaluate on (doc 8).

**Inference** - Running a trained model to get answers. Much cheaper than training.

**Instruct model** - A base model further trained to follow instructions. Usually STRATUM's starting point.

**Loss** - A single number measuring how wrong the model's predictions are. Training lowers it.

**Loss mask** - Marking prompt tokens as "ignore" (`-100`) so loss is computed only on the response. Getting this wrong silently ruins fine-tuning.

**LoRA (Low-Rank Adaptation)** - Building an adapter as two small matrices A and B whose product is the adjustment added to a frozen weight. The basis of strata.

**Merge / fuse** - Combining strata into one model by (weighted) addition of their deltas.

**Momentum** - A smoothed average of a parameter's recent gradients. The single note Muon keeps.

**MPS (Metal Performance Shaders)** - How PyTorch uses the GPU built into Apple silicon Macs. STRATUM picks it up automatically when there is no CUDA device.

**Muon (MomentUm Orthogonalized by Newton-schulz)** - STRATUM's optimizer for weight matrices. Rebalances each matrix's update so every direction shares (orthogonalization) and keeps one note per parameter - half AdamW's optimizer memory, with a step advantage that grows with model size (doc 4's race).

**Newton-Schulz iteration** - The short routine (a few matrix multiplies) Muon uses to orthogonalize an update without computing singular values directly.

**OCR (optical character recognition)** - Software that reads pictures of text, e.g. scanned pages, into actual text. A scanned PDF needs OCR (or a vision teacher) before it can be ingested.

**Optimizer** - The algorithm that turns gradients into parameter updates. SGD, AdamW, Muon.

**Orthogonalization** - Muon's core move: flattening a matrix update's singular values toward 1 so no direction dominates.

**Parameter / weights** - The billions of numbers inside a model. Training adjusts them.

**PII (personally identifiable information)** - Names, emails, phone numbers, identifiers. `stratum corpus ingest --redact` scrubs the obvious kinds as a second net - a regulated deployment runs its own PII pipeline before ingest.

**PEFT (Parameter-Efficient Fine-Tuning)** - The family of methods (including LoRA) that train a small add-on instead of the whole model. Also the Hugging Face library STRATUM uses.

**Plan / preflight** - `stratum plan` estimates each stratum's training memory against this machine's hardware and says fits, tight, or does not fit, with fixes or a rented-hardware handoff. `stratum stack` runs the same check before training.

**Eval gate** - A test set plus a minimum score in a recipe's `evals` section. The build fails if the merged model scores below the bar, making a recipe a self-verifying build spec.

**Pretraining** - The original, enormously expensive training on trillions of tokens. You don't do this - you start from a pretrained model.

**Quantization** - Storing numbers with fewer bits (e.g. 4 vs 16) to save memory, at small quality cost. Used on the frozen base to fit a laptop.

**QLoRA** - LoRA plus quantization of the frozen base. Fits multi-billion-parameter training on a laptop.

**RAG (retrieval-augmented generation)** - Answering from documents by looking the relevant passages up at question time and pasting them into the model's prompt. The right tool for knowledge (exact, citable, current), where fine-tuning is the right tool for skills - doc 14 draws the line.

**Rank (r)** - The size of a LoRA adapter's small matrices. It controls how complex an adjustment it can represent. Too low and it plateaus. 16 for style, 32-64 for knowledge.

**Recipe** - A YAML file describing a whole STRATUM build: which strata to train with which settings, how to fuse them, and which eval gates the result must pass. Run with `stratum stack`, checked against your hardware with `stratum plan`.

**SGD (Stochastic Gradient Descent)** - The simplest optimizer: step along the gradient. Crude but foundational.

**SLM (Small Language Model)** - A language model small enough to train and serve on modest hardware, typically under about 10B parameters. What STRATUM builds.

**Singular values** - Numbers describing how strongly a matrix update pushes along each independent direction. Muon flattens them toward 1.

**System prompt** - An instruction message placed before the user's input ("You are a precise assistant for invoice processing.") that steers every answer. Set with `--system` or the recipe's `system` key - and use the SAME one at training and serving time, or behavior shifts.

**Stratum / strata** - STRATUM's word for a skill adapter / adapters. Small, separate, mergeable - the pieces you build from.

**stratum_card.json** - The provenance file saved with each stratum: base model, skill file and its SHA-256 fingerprint, every setting, final loss, and build time. Used to verify merge compatibility and as an audit trail. The merged model's `stratum_merge.json` carries all input cards forward.

**Transformer** - The stack design used by every model in this project: a few dozen layers, each pairing an attention step with plain multiply-through-grids processing. STRATUM adjusts the dials inside the grids and never changes the design.

**Token** - A chunk of text (~3/4 word) the model reads and writes in. Models have a fixed vocabulary of tokens.

**Thinking model** - A model (Qwen3 among them) whose chat template lets it write reasoning inside `<think>` tags before answering. STRATUM disables thinking during training and inference so a specialized model answers directly, and strips any think blocks that appear anyway.

**TIES** - A merge method that trims each stratum to its most important adjustments and resolves direction conflicts by majority vote. From Yadav et al. (2023).

**Vision-language model (VLM)** - A model that reads images the way a language model reads text. STRATUM uses one as a vision teacher to extract image content into the corpus - fine-tuning a VLM itself is a different pipeline and out of current scope.

**VRAM (Video RAM)** - Graphics card memory. The binding constraint on what you can train.
