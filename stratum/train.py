"""
Train one skill stratum: a LoRA adapter optimized with Muon (or AdamW).

This is the step that runs on your laptop. Pipeline:
  1. load frozen base (4-bit quantized if requested, to fit small VRAM)
  2. attach a LoRA adapter (the trainable stratum)
  3. train only the adapter, batching with a proper loss mask
  4. save the adapter + a stratum_card.json recording provenance

Everything is deterministic given a seed. See docs/06-training.md.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .data import file_sha256, load_jsonl, make_batches
from .muon import Muon, split_params_for_muon


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_tile(
    skill_path: str,
    out_dir: str,
    base_model: str = "Qwen/Qwen3-1.7B",
    rank: int = 16,
    lora_alpha: int | None = None,
    lr: float = 2e-2,
    adamw_lr: float = 1e-3,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_len: int = 1024,
    optimizer: str = "muon",
    system: str | None = None,
    load_4bit: bool = True,
    seed: int = 42,
    log_every: int = 10,
):
    """Train a single stratum and save it to out_dir. Returns the final avg loss."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    set_seed(seed)
    lora_alpha = lora_alpha or rank * 2

    rows = load_jsonl(skill_path, required_keys=("prompt", "response"))
    print(f"Loaded {len(rows)} training pairs from {skill_path}")

    # Friendly preflight: catch a bad model id before the cryptic download error.
    from .hf_utils import resolve_model_hint
    hint = resolve_model_hint(base_model)
    if hint:
        print(f"Note: {hint}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except Exception as e:
        raise RuntimeError(
            f"Could not load base model '{base_model}'.\n"
            f" - Check the id (e.g. 'Qwen/Qwen3-1.7B') or use a local folder path.\n"
            f" - Gated model? Run `huggingface-cli login`.\n"
            f" - No internet? Set HF_HUB_OFFLINE=1 and use a local path.\n"
            f" - Run `stratum doctor` to check Hugging Face readiness.\n"
            f"Original error: {e}"
        ) from e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(torch_dtype=torch.bfloat16)
    if load_4bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print("Loading base in 4-bit (QLoRA).")
        except Exception as e:
            print(f"4-bit unavailable ({e}) - loading in bf16 (needs more VRAM).")

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    # Prepare a quantized model for training if needed (enables gradient flow).
    if load_4bit and "quantization_config" in model_kwargs:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    from .hf_utils import pick_device
    device = pick_device()
    if "quantization_config" not in model_kwargs:
        model.to(device)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        # The generation cache is useless during training and fights with
        # checkpointing. Turning it off silences the warning and saves memory.
        model.config.use_cache = False

    # Build optimizer(s).
    if optimizer == "muon":
        muon_p, adamw_p = split_params_for_muon(model)
        opts = [Muon(muon_p, lr=lr, weight_decay=0.01)]
        if adamw_p:
            opts.append(torch.optim.AdamW(adamw_p, lr=adamw_lr))
        print(f"Optimizer: Muon on {len(muon_p)} matrices, "
              f"AdamW on {len(adamw_p)} other params.")
    elif optimizer == "adamw":
        trainable = [p for p in model.parameters() if p.requires_grad]
        opts = [torch.optim.AdamW(trainable, lr=adamw_lr)]
        print(f"Optimizer: AdamW on {len(trainable)} trainable params.")
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Use 'muon' or 'adamw'.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(epochs_completed, loss_so_far):
        # The adapter is a few megabytes, so saving after every epoch is
        # nearly free - and it means a long run that gets killed (power, OOM,
        # an impatient operator) leaves the last full epoch usable instead
        # of nothing. On a slow CPU run this is the difference between
        # losing an hour and losing nothing.
        model.save_pretrained(str(out_path))
        tokenizer.save_pretrained(str(out_path))
        card = {
            "stratum_name": out_path.name,
            "base_model": base_model,
            "rank": rank,
            "lora_alpha": lora_alpha,
            "optimizer": optimizer,
            "lr": lr if optimizer == "muon" else adamw_lr,
            "epochs": epochs,
            "epochs_completed": epochs_completed,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "max_len": max_len,
            "load_4bit": bool(load_4bit and "quantization_config" in model_kwargs),
            "skill_file": skill_path,
            "skill_sha256": file_sha256(skill_path),
            "num_pairs": len(rows),
            "final_loss": loss_so_far,
            "seed": seed,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stratum_version": 1,
        }
        (out_path / "stratum_card.json").write_text(json.dumps(card, indent=2),
                                                    encoding="utf-8")

    # Training loop with gradient accumulation.
    t0 = time.time()
    final_loss = None
    for epoch in range(epochs):
        random.shuffle(rows)
        total, n_steps = 0.0, 0
        batches = list(make_batches(tokenizer, rows, system, max_len, batch_size))
        for opt in opts:
            opt.zero_grad()
        for step, (input_ids, attn, labels) in enumerate(batches):
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)

            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            # Safety net: skip a batch whose loss is non-finite rather than let
            # it poison training. Rare, but possible with tiny data or bad rows.
            if not torch.isfinite(out.loss):
                print(f" (skipped non-finite loss at epoch {epoch+1} step {step+1})")
                for opt in opts:
                    opt.zero_grad()
                continue
            loss = out.loss / grad_accum
            loss.backward()
            total += out.loss.item()
            n_steps += 1

            if (step + 1) % grad_accum == 0 or (step + 1) == len(batches):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                for opt in opts:
                    opt.step()
                    opt.zero_grad()

            if (step + 1) % log_every == 0:
                print(f" epoch {epoch+1} step {step+1}/{len(batches)} "
                      f"loss {out.loss.item():.4f}")

        avg = total / max(n_steps, 1)
        final_loss = avg
        save_checkpoint(epoch + 1, avg)
        print(f"epoch {epoch+1}/{epochs} avg loss {avg:.4f} "
              f"({time.time()-t0:.0f}s elapsed) - checkpoint saved")

    print(f"\nStratum saved to {out_dir}")
    print(f" base: {base_model} rank: {rank} final loss: {final_loss:.4f}")
    return final_loss
