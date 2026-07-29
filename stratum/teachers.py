"""
Teacher backends for data distillation (`stratum teacher-gen`).

A "teacher" is any callable str -> str that answers a prompt well. STRATUM ships
four backends and stays vendor-neutral so you can use whatever you have access to:

  hf a local Hugging Face model (downloaded from the Hub or a local path)
  openai the OpenAI API (needs OPENAI_API_KEY and `pip install openai`)
  anthropic the Anthropic API (needs ANTHROPIC_API_KEY and `pip install anthropic`)
  echo a no-op teacher for testing the pipeline without any model

Every backend fails LOUDLY and CLEARLY if something is missing - a normal
enterprise developer (Java/.NET/Python) should never be left guessing.
"""
from __future__ import annotations

import os


def get_teacher(backend: str, model: str | None = None):
    """Return a callable str->str for the requested teacher backend."""
    backend = backend.lower()
    if backend == "hf":
        return _hf_teacher(model or "Qwen/Qwen3-4B")
    if backend == "openai":
        return _openai_teacher(model or "gpt-4o-mini")
    if backend == "anthropic":
        return _anthropic_teacher(model or "claude-3-5-sonnet-latest")
    if backend == "echo":
        return lambda prompt: "(echo teacher - replace with a real backend)"
    raise ValueError(f"Unknown teacher backend '{backend}'. "
                     f"Choose from: hf, openai, anthropic, echo.")


def _hf_teacher(model_name: str):
    """A local Hugging Face model as teacher.

    Downloads from the Hub on first use (needs internet + `huggingface_hub`),
    or loads from a local directory path. Runs on GPU if available.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "The 'hf' teacher needs transformers + torch. "
            "Install with: pip install transformers torch"
        ) from e

    print(f"Loading Hugging Face teacher: {model_name}")
    print("(first run downloads the model - this can take a while)")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    except Exception as e:
        raise RuntimeError(
            f"Could not load teacher '{model_name}'. Common causes:\n"
            f" - No internet, or the model id is misspelled.\n"
            f" - The model is gated: run `huggingface-cli login` first.\n"
            f" - Out of memory: pick a smaller teacher.\n"
            f"Original error: {e}"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def teacher(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
        except Exception:
            text = prompt
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=512, do_sample=False,
                                 temperature=None, top_p=None,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    return teacher


def _openai_teacher(model_name: str):
    """OpenAI API as teacher. Needs OPENAI_API_KEY and `pip install openai`."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it first:\n"
            " export OPENAI_API_KEY=sk-...\n"
            "Or use `--teacher hf` for a local model instead."
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai") from e

    client = OpenAI()

    def teacher(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content

    return teacher


def _anthropic_teacher(model_name: str):
    """Anthropic API as teacher. Needs ANTHROPIC_API_KEY and `pip install anthropic`."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Export it first:\n"
            " export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Or use `--teacher hf` for a local model instead."
        )
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("pip install anthropic") from e

    client = anthropic.Anthropic()

    def teacher(prompt: str) -> str:
        resp = client.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    return teacher
