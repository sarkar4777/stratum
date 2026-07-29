"""
Distillation for STRATUM.

Distillation = teaching a small "student" model to imitate a large "teacher"
model. Instead of (or in addition to) training on fixed human-written responses,
the student learns from the teacher's richer output.

STRATUM supports two flavors, from simplest to most powerful:

  1. DATA distillation (offline, recommended default)
     Use a big teacher (a frontier API model, or a large local model) to GENERATE
     the training pairs, then train a normal stratum on them with train.py.
     Simple, robust, and the teacher can even be a closed API. See
     generate_dataset_from_teacher() below and docs/05-distillation.md.

  2. LOGIT distillation (online, advanced)
     Run teacher and student on the same text at once and train the student to
     match the teacher's full next-token probability distribution (its "soft
     labels"), not just the single correct token. Richer signal, but requires
     BOTH models to share a tokenizer and to fit in memory together. See
     distill_tile() below.

Most users should start with data distillation - it gives 90% of the benefit
with none of the memory pain. Logit distillation is here for when you have a
capable local teacher and want to squeeze out the last bit of quality.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. DATA DISTILLATION - teacher writes the training data
# --------------------------------------------------------------------------

def build_teacher_prompt(task_instruction: str, seed_input: str) -> str:
    """Compose a prompt asking the teacher to produce a high-quality response.

    You supply a task instruction (what the skill is) and a seed input (the
    concrete case). The teacher's answer becomes one training pair's response.
    """
    return (
        f"{task_instruction}\n\n"
        f"Input:\n{seed_input}\n\n"
        f"Provide only the ideal response, with no preamble or explanation."
    )


def generate_dataset_from_teacher(
    seed_inputs: list[str],
    task_instruction: str,
    teacher_fn,
    out_path: str,
) -> str:
    """Generate a training JSONL by asking a teacher for each seed input.

    Args:
        seed_inputs: the concrete inputs (documents, tickets, questions...).
        task_instruction: what the skill should do, in one line.
        teacher_fn: a callable str->str. YOU provide this. It can wrap any
            teacher: an OpenAI/Anthropic API call, a local big model, anything.
            STRATUM stays vendor-neutral by taking this function from you.
        out_path: where to write the resulting {"prompt","response"} JSONL.

    Returns out_path. See docs/05-distillation.md for a ready teacher_fn.
    """
    rows = []
    for i, seed in enumerate(seed_inputs, 1):
        prompt_to_teacher = build_teacher_prompt(task_instruction, seed)
        response = teacher_fn(prompt_to_teacher).strip()
        # The STUDENT will be trained on the clean task prompt -> teacher answer.
        student_prompt = f"{task_instruction}\n\nInput:\n{seed}"
        rows.append({"prompt": student_prompt, "response": response})
        if i % 10 == 0:
            print(f" generated {i}/{len(seed_inputs)} pairs")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} distilled pairs to {out_path}")
    return out_path


# --------------------------------------------------------------------------
# 2. LOGIT DISTILLATION - student matches teacher's soft probabilities
# --------------------------------------------------------------------------

def distillation_loss(student_logits, teacher_logits, labels,
                      temperature: float = 2.0, alpha: float = 0.5):
    """Combine two losses for logit distillation.

    - Soft loss: student's distribution should match the teacher's softened
      distribution (KL divergence). This is the rich signal - it teaches the
      student the teacher's UNCERTAINTY (e.g. "88 is likely but 89 plausible").
    - Hard loss: ordinary next-token loss against the true labels.

    temperature > 1 softens both distributions so the student learns from the
    teacher's smaller, informative probabilities, not just its top pick.
    alpha weights soft vs hard (0.5 = equal).

    Loss is computed only where labels != -100 (the response tokens).
    """
    # Shift so position t predicts token t+1 (standard causal setup).
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :]
    lab = labels[:, 1:]

    mask = (lab != -100)
    if mask.sum() == 0:
        return student_logits.sum() * 0.0 # nothing to learn from

    s = s[mask] # [N, vocab]
    t = t[mask] # [N, vocab]
    lab = lab[mask] # [N]

    # Soft loss: KL( teacher || student ), both softened by temperature.
    T = temperature
    soft = F.kl_div(
        F.log_softmax(s / T, dim=-1),
        F.softmax(t / T, dim=-1),
        reduction="batchmean",
    ) * (T * T) # scale restores gradient magnitude

    # Hard loss: standard cross-entropy against the real tokens.
    hard = F.cross_entropy(s, lab)

    return alpha * soft + (1 - alpha) * hard


def distill_tile(
    skill_path: str,
    out_dir: str,
    student_model: str = "Qwen/Qwen3-1.7B",
    teacher_model: str = "Qwen/Qwen3-4B",
    rank: int = 16,
    lr: float = 2e-2,
    epochs: int = 3,
    batch_size: int = 2,
    max_len: int = 1024,
    temperature: float = 2.0,
    alpha: float = 0.5,
    system: str | None = None,
    seed: int = 42,
):
    """Train a student stratum to imitate a teacher via logit distillation.

    Requires teacher and student to SHARE A TOKENIZER (same model family, e.g.
    both Qwen3). The teacher runs frozen in eval mode; only the student's LoRA
    adapter trains. Both models must fit in memory together - use a small
    student and a mid-size teacher, or prefer data distillation above.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    from .data import load_jsonl, make_batches
    from .muon import Muon, split_params_for_muon
    from .train import set_seed

    set_seed(seed)
    rows = load_jsonl(skill_path, required_keys=("prompt", "response"))
    print(f"Distilling {student_model} from teacher {teacher_model}")
    print(f"Loaded {len(rows)} pairs from {skill_path}")

    tokenizer = AutoTokenizer.from_pretrained(student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Teacher: frozen, eval mode.
    teacher = AutoModelForCausalLM.from_pretrained(teacher_model, torch_dtype=torch.bfloat16)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student: LoRA adapter trains.
    student = AutoModelForCausalLM.from_pretrained(student_model, torch_dtype=torch.bfloat16)
    lora_cfg = LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
                          target_modules="all-linear", task_type="CAUSAL_LM")
    student = get_peft_model(student, lora_cfg)
    student.to(device).train()
    student.print_trainable_parameters()

    # Verify shared vocab (a common footgun).
    if teacher.config.vocab_size != student.config.vocab_size:
        raise ValueError(
            f"Teacher vocab ({teacher.config.vocab_size}) != student vocab "
            f"({student.config.vocab_size}). Logit distillation needs a shared "
            f"tokenizer - use models from the same family, or use data "
            f"distillation instead (generate_dataset_from_teacher)."
        )

    muon_p, adamw_p = split_params_for_muon(student)
    opts = [Muon(muon_p, lr=lr, weight_decay=0.01)]
    if adamw_p:
        opts.append(torch.optim.AdamW(adamw_p, lr=1e-3))

    for epoch in range(epochs):
        total = 0.0
        batches = list(make_batches(tokenizer, rows, system, max_len, batch_size))
        for input_ids, attn, labels in batches:
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)

            with torch.no_grad():
                t_logits = teacher(input_ids=input_ids, attention_mask=attn).logits
            s_logits = student(input_ids=input_ids, attention_mask=attn).logits

            loss = distillation_loss(s_logits, t_logits, labels, temperature, alpha)
            if not torch.isfinite(loss):
                for opt in opts:
                    opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0)
            for opt in opts:
                opt.step()
                opt.zero_grad()
            total += loss.item()
        print(f"epoch {epoch+1}/{epochs} avg distill loss {total/max(len(batches),1):.4f}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "stratum_card.json").write_text(json.dumps({
        "stratum_name": out.name, "base_model": student_model,
        "teacher_model": teacher_model, "rank": rank, "method": "logit_distillation",
        "temperature": temperature, "alpha": alpha, "skill_file": skill_path,
        "num_pairs": len(rows), "seed": seed, "stratum_version": 1,
    }, indent=2))
    print(f"\nDistilled stratum saved to {out_dir} (base: {student_model})")
