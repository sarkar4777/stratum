# 11 - Glossary

*Every term in STRATUM, with full form and plain definition.*

**Adapter** - A small trainable add-on attached to a frozen model that adjusts it for one skill. In STRATUM, a "stratum." Often under 1% of the model's size.

**AdamW (Adam with decoupled Weight decay)** - The standard optimizer. Keeps two running notes per parameter (momentum, variance) - effective but memory-heavy. STRATUM uses Muon for weight matrices and a small AdamW for the rest.

**Base model** - The pretrained model you start from and specialize. All strata you merge must share one base.

**Batch** - Several training examples processed together for GPU efficiency.

**bf16 (brain float 16)** - Storing each number in 2 bytes - the common training format.

**Catastrophic forgetting / over-specialization** - When training hard on a narrow skill makes the model worse at everything else. Guard by limiting epochs and checking general ability.

**DARE** - A merge method that randomly drops most of each stratum's small adjustments and rescales the rest, reducing crosstalk when fusing many strata. From Yu et al. (2023). STRATUM seeds the randomness so the same merge always gives the same model.

**Delta** - A stratum's weight adjustment, `scaling x (B @ A)`, added onto a base weight during merging.

**Distillation** - Training a small "student" model to imitate a large "teacher" model, so the student captures the teacher's skill cheaply. STRATUM supports data distillation (teacher writes the training pairs) and logit distillation (student matches the teacher's probability distribution).

**Data distillation** - The simple flavor: a teacher model generates the training pairs, which then train a normal stratum. Works with any teacher, including closed APIs.

**Logit distillation** - The advanced flavor: teacher and student run together and the student is trained to match the teacher's full softened distribution (soft labels). Needs a shared tokenizer.

**Soft labels** - A teacher's full probability distribution over next tokens (e.g. billing 88%, account 9%, ...), which carries richer information than the single correct answer (the hard label). The signal logit distillation learns from.

**KL divergence (Kullback-Leibler divergence)** - A measure of how different two probability distributions are. Logit distillation minimizes it to pull the student's distribution toward the teacher's.

**Temperature (distillation)** - A factor that softens probability distributions so the student learns from the teacher's smaller, informative probabilities, not just its top pick. Higher = softer.

**Teacher / student** - In distillation, the large capable model being imitated (teacher) and the small model being trained (student).

**Epoch** - One full pass over the training data.

**Fine-tuning** - Further training of a pretrained model to specialize it. Cheap. What STRATUM does.

**Gradient** - For each parameter, the direction that would lower the loss. Computed in the backward pass.

**Gradient accumulation** - Summing gradients over several small batches before one update, to get a large effective batch within limited memory.

**Gradient clipping** - Capping gradient size before an update to prevent instability.

**Inference** - Running a trained model to get answers. Much cheaper than training.

**Instruct model** - A base model further trained to follow instructions. Usually STRATUM's starting point.

**Loss** - A single number measuring how wrong the model's predictions are. Training lowers it.

**Loss mask** - Marking prompt tokens as "ignore" (`-100`) so loss is computed only on the response. Getting this wrong silently ruins fine-tuning.

**LoRA (Low-Rank Adaptation)** - Building an adapter as two small matrices A and B whose product is the adjustment added to a frozen weight. The basis of strata.

**Merge / fuse** - Combining strata into one model by (weighted) addition of their deltas.

**Momentum** - A smoothed average of a parameter's recent gradients. The single note Muon keeps.

**Muon (MomentUm Orthogonalized by Newton-schulz)** - STRATUM's optimizer for weight matrices. Rebalances each matrix's update so every direction shares (orthogonalization), keeps one note per parameter, needs fewer steps.

**Newton-Schulz iteration** - The short routine (a few matrix multiplies) Muon uses to orthogonalize an update without computing singular values directly.

**Optimizer** - The algorithm that turns gradients into parameter updates. SGD, AdamW, Muon.

**Orthogonalization** - Muon's core move: flattening a matrix update's singular values toward 1 so no direction dominates.

**Parameter / weights** - The billions of numbers inside a model. Training adjusts them.

**PEFT (Parameter-Efficient Fine-Tuning)** - The family of methods (including LoRA) that train a small add-on instead of the whole model. Also the Hugging Face library STRATUM uses.

**Pretraining** - The original, enormously expensive training on trillions of tokens. You don't do this - you start from a pretrained model.

**Quantization** - Storing numbers with fewer bits (e.g. 4 vs 16) to save memory, at small quality cost. Used on the frozen base to fit a laptop.

**QLoRA** - LoRA plus quantization of the frozen base. Fits multi-billion-parameter training on a laptop.

**Rank (r)** - The size of a LoRA adapter's small matrices. It controls how complex an adjustment it can represent. Too low and it plateaus. 16 for style, 32-64 for knowledge.

**Recipe** - A YAML file describing a whole STRATUM build (which strata to train, how to fuse). Run with `stratum stack`.

**SGD (Stochastic Gradient Descent)** - The simplest optimizer: step along the gradient. Crude but foundational.

**Singular values** - Numbers describing how strongly a matrix update pushes along each independent direction. Muon flattens them toward 1.

**Stratum / strata** - STRATUM's word for a skill adapter / adapters. Small, separate, mergeable - the pieces you build from.

**stratum_card.json** - The provenance file saved with each stratum: base model, skill file, settings, final loss. Used to verify merge compatibility and as an audit trail.

**Token** - A chunk of text (~3/4 word) the model reads and writes in. Models have a fixed vocabulary of tokens.

**Thinking model** - A model (Qwen3 among them) whose chat template lets it write reasoning inside `<think>` tags before answering. STRATUM disables thinking during training and inference so a specialized model answers directly, and strips any think blocks that appear anyway.

**TIES** - A merge method that trims each stratum to its most important adjustments and resolves direction conflicts by majority vote. From Yadav et al. (2023).

**VRAM (Video RAM)** - Graphics card memory. The binding constraint on what you can train.
