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


def resolve_model_hint(model_name: str) -> str:
    """Return a friendly hint if a model id looks like a common mistake."""
    if "/" not in model_name and not os.path.isdir(model_name):
        return (f"'{model_name}' has no namespace and isn't a local folder. "
                f"Hub ids look like 'Qwen/Qwen3-1.7B'. Did you mean a full id "
                f"or a local path?")
    return ""
