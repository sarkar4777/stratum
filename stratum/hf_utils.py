"""
Hugging Face helpers - make model loading dumb-proof.

Enterprise developers new to this hit the same three walls: no internet on a
locked-down machine, gated models needing a login, and cryptic download errors.
This module gives one place to check readiness and turn obscure failures into
plain instructions.
"""
from __future__ import annotations

import os


def check_hf_ready(verbose: bool = True) -> dict:
    """Check whether Hugging Face model downloads will work. Returns a status dict."""
    status = {"hub_installed": False, "logged_in": False, "online": False, "token": None}

    try:
        import huggingface_hub # noqa
        status["hub_installed"] = True
    except ImportError:
        if verbose:
            print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return status

    # Token (needed for gated models like some Llama/Qwen releases).
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    status["token"] = bool(token)
    status["logged_in"] = bool(token)

    # Connectivity.
    try:
        from huggingface_hub import HfApi
        HfApi().whoami(token=token) if token else HfApi().model_info("gpt2")
        status["online"] = True
    except Exception:
        status["online"] = False

    if verbose:
        print("Hugging Face readiness:")
        print(f" library installed: {'yes' if status['hub_installed'] else 'NO'}")
        print(f" logged in (token): {'yes' if status['logged_in'] else 'no (fine for open models)'}")
        print(f" can reach the Hub: {'yes' if status['online'] else 'NO - check internet/proxy'}")
        if not status["online"]:
            print("\n If you're on a locked-down network:")
            print(" - set HF_HUB_OFFLINE=1 and point --base at a local model folder, OR")
            print(" - pre-download on a connected machine with "
                  "`huggingface-cli download <model>` and copy the folder over.")
        if not status["logged_in"]:
            print("\n For gated models (some Llama/Qwen): run `huggingface-cli login`.")
    return status


def pick_device() -> str:
    """The best device the installed PyTorch can actually use.

    cuda (NVIDIA) first, then mps (Apple silicon), then cpu. Every training
    and inference path calls this, so a Mac's GPU is used automatically
    instead of silently falling back to CPU.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def detect_hidden_nvidia_gpu() -> str | None:
    """Spot an NVIDIA GPU that the installed PyTorch cannot see.

    The single most common cause of "why is my GPU not being used" is a
    perfectly good NVIDIA card sitting behind a CPU-only PyTorch build -
    torch.cuda.is_available() says False and everything silently runs slow.
    This asks the driver directly (nvidia-smi) so doctor can tell the user
    the actual fix instead of "no GPU".
    """
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    name = result.stdout.strip().splitlines()
    return name[0].strip() if name else None


def load_for_inference(model_dir: str):
    """Load a model directory for generation, whatever kind it is.

    Two kinds of directory come out of STRATUM:
      - a merged model (from `stratum merge`): a full standalone model
      - a single stratum (from `stratum train`): just an adapter, which needs
        its base model loaded first and the adapter attached on top

    This detects which one it got, so `stratum eval` and `stratum chat` work
    on both - you can score a stratum on its own before ever merging it.
    Returns (model, tokenizer).
    """
    import json
    import torch
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_dir)
    adapter_cfg = path / "adapter_config.json"
    if adapter_cfg.exists():
        # A stratum. Find its base from the card (preferred - it is STRATUM's
        # provenance record) or from what PEFT wrote in the adapter config.
        base = None
        card_path = path / "stratum_card.json"
        if card_path.exists():
            base = json.loads(card_path.read_text(encoding="utf-8")).get("base_model")
        if not base:
            base = json.loads(adapter_cfg.read_text(encoding="utf-8")).get(
                "base_model_name_or_path")
        if not base:
            raise ValueError(
                f"{model_dir} is an adapter but neither stratum_card.json nor "
                f"adapter_config.json names its base model."
            )
        print(f"{model_dir} is a single stratum - loading base {base} and attaching it.")
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    return model, tokenizer


def resolve_model_hint(model_name: str) -> str:
    """Return a friendly hint if a model id looks like a common mistake."""
    if "/" not in model_name and not os.path.isdir(model_name):
        return (f"'{model_name}' has no namespace and isn't a local folder. "
                f"Hub ids look like 'Qwen/Qwen3-1.7B'. Did you mean a full id "
                f"or a local path?")
    return ""
