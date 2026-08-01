"""
Hugging Face helpers - make model loading dumb-proof.

Enterprise developers new to this hit the same four walls: no internet on a
locked-down machine, gated models needing a login, cryptic download errors, and
a PyTorch too old for the transformers that pip installed next to it. This
module gives one place to check readiness and turn obscure failures into plain
instructions.
"""
from __future__ import annotations

import os


def check_torch_stack(verbose: bool = False) -> dict:
    """Check that PyTorch, transformers and peft can work together.

    transformers refuses to use a PyTorch older than its minimum: it quietly
    disables its own torch half, and the next import that reaches modelling
    code dies with a bare `NameError: name 'torch' is not defined`. That
    traceback names no version and no package, so the real cause - two
    libraries that pip was happy to install side by side but that do not
    match - is invisible.

    Nothing here compares version numbers by hand: the minimum torch moves
    with every transformers release, so only the installed transformers knows
    the answer. We ask it. Returns a status dict; `ok` False means every
    command that loads a model will fail.
    """
    from importlib.metadata import PackageNotFoundError, version

    def installed(pkg: str) -> str | None:
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    status = {p: installed(p) for p in
              ("torch", "transformers", "peft", "accelerate")}
    status.update(ok=False, problem=None, fix=[])

    missing = [p for p in ("torch", "transformers", "peft") if not status[p]]
    if missing:
        status["problem"] = f"Not installed: {', '.join(missing)}."
        status["fix"] = ["pip install -e .   (from the STRATUM folder)"]
    else:
        # Ask transformers whether it accepted this torch. Its own warning
        # says the same thing less usefully, so keep it out of the output.
        import logging
        logging.disable(logging.WARNING)
        try:
            from transformers.utils import is_torch_available
            status["ok"] = bool(is_torch_available())
        except Exception as e:
            status["ok"] = False
            status["problem"] = f"transformers is installed but unusable: {e}"
        finally:
            logging.disable(logging.NOTSET)

        if not status["ok"] and not status["problem"]:
            status["problem"] = (
                f"transformers {status['transformers']} will not use PyTorch "
                f"{status['torch']} - it requires a newer one, so it has "
                f"disabled PyTorch entirely.")
            status["fix"] = _torch_stack_fix()

    if verbose:
        print("Library versions:")
        for pkg in ("torch", "transformers", "peft", "accelerate"):
            print(f" {pkg}: {status[pkg] or 'NOT INSTALLED'}")
        if status["ok"]:
            print(" these versions work together: yes")
        else:
            print(f"\n PROBLEM: {status['problem']}")
            print(" Every command that loads a model will fail with"
                  ' "NameError: name \'torch\' is not defined".')
            print("\n Fix it with:")
            for line in status["fix"]:
                print(f"   {line}")
    return status


def _torch_stack_fix() -> list[str]:
    """The commands that repair a mismatched stack on *this* machine.

    Which advice is right depends on whether a newer PyTorch exists for the
    platform at all. It does not on Intel Macs - PyTorch stopped publishing
    macOS x86_64 builds after 2.2.2 - so telling that user to upgrade torch
    sends them in a circle. There the only way out is down: older libraries.
    """
    import platform

    pin = 'pip install "transformers<5"'
    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        return [
            pin,
            "",
            "(PyTorch publishes no macOS-Intel build after 2.2.2, so upgrading",
            " torch is not an option on this machine - pin the libraries",
            " instead. transformers 4.x supports PyTorch 2.2.)",
        ]
    return [
        "pip install -U torch          # preferred: newer torch, newest libraries",
        f"{pin}    # or: keep this torch, older libraries",
    ]


def require_torch_stack() -> None:
    """Stop with a readable message when the installed libraries cannot work.

    Called before any command that loads a model, so a version mismatch costs
    the user five lines of explanation instead of a 20-frame traceback.
    """
    status = check_torch_stack()
    if status["ok"]:
        return
    lines = [status["problem"], ""]
    lines += ["Fix it with:"] + [f"  {f}" for f in status["fix"]]
    lines += ["", "Then re-run `stratum doctor` to confirm."]
    raise SystemExit("\n".join(lines))


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


def encode_for_generation(tokenizer, text: str, device: str):
    """Tokenize a prompt into exactly what a causal model's generate() accepts.

    Tokenizers may return extras - token_type_ids is the common one - that
    causal models have no argument for. transformers refuses unknown
    generate() kwargs ("The following model_kwargs are not used by the
    model"), so passing the tokenizer's output through wholesale breaks on
    those tokenizers. Keep the two tensors every causal model wants.
    """
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return {k: v.to(device) for k, v in ids.items()
            if k in ("input_ids", "attention_mask")}


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
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    return model, tokenizer


def resolve_model_hint(model_name: str) -> str:
    """Return a friendly hint if a model id looks like a common mistake."""
    if "/" not in model_name and not os.path.isdir(model_name):
        return (f"'{model_name}' has no namespace and isn't a local folder. "
                f"Hub ids look like 'Qwen/Qwen3-1.7B'. Did you mean a full id "
                f"or a local path?")
    return ""
