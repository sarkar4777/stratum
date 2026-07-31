"""
Vision teachers: turn images into text the corpus pipeline can use.

A corpus is rarely text alone - it carries diagrams, scanned pages, charts,
photos of equipment. A vision teacher is a callable image-path -> text that
reads one image and writes down everything in it: transcribed text, table
contents, what a diagram shows. That text then flows through the normal
pipeline (chunks, pairs, strata) like any document.

Be clear with the client about what this buys: the FINISHED model is still a
text model. It knows what was IN the images at build time, but it cannot look
at a new image at inference time - that needs a vision-language model, which
is a different training pipeline and out of STRATUM's current scope (the
roadmap note in docs/14 says so plainly).

Backends mirror stratum.teachers:

  hf         a local vision-language model - nothing leaves your machine,
             the right choice for regulated corpora
  anthropic  the Anthropic API - sends every image to the API
  openai     the OpenAI API - sends every image to the API
  gemini     the Google Gemini API - sends every image to the API
  echo       a no-op for testing the pipeline without any model
"""
from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

EXTRACT_INSTRUCTION = (
    "Write down everything in this image for a text-only archive. Transcribe "
    "all visible text exactly. Describe tables row by row. For diagrams and "
    "charts, state what they show, the labels, and the values. For photos, "
    "describe what is depicted and any identifying labels. Be complete and "
    "factual, with no commentary."
)


def get_vision_teacher(backend: str, model: str | None = None,
                       instruction: str | None = None):
    """Return a callable image-path -> extracted text."""
    backend = backend.lower()
    instruction = instruction or EXTRACT_INSTRUCTION
    if backend == "hf":
        return _hf_vision_teacher(model or "Qwen/Qwen2.5-VL-3B-Instruct", instruction)
    if backend == "anthropic":
        return _anthropic_vision_teacher(model or "claude-opus-5", instruction)
    if backend == "openai":
        return _openai_vision_teacher(model or "gpt-4o-mini", instruction)
    if backend == "gemini":
        return _gemini_vision_teacher(model or "gemini-2.5-flash", instruction)
    if backend == "echo":
        return lambda path: f"(echo vision teacher - contents of {Path(path).name})"
    raise ValueError(f"Unknown vision backend '{backend}'. "
                     f"Choose from: hf, anthropic, openai, gemini, echo.")


def _hf_vision_teacher(model_name: str, instruction: str):
    """A local vision-language model. Images never leave the machine."""
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as e:
        raise ImportError(
            "The 'hf' vision teacher needs transformers and torch, plus "
            "pillow for image loading: pip install 'stratum-slm[corpus]'"
        ) from e

    print(f"Loading vision teacher: {model_name}")
    print("(first run downloads the model - this can take a while)")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16)
    except Exception as e:
        raise RuntimeError(
            f"Could not load vision teacher '{model_name}'. Common causes:\n"
            f" - No internet, or the model id is misspelled.\n"
            f" - Out of memory: pick a smaller vision model.\n"
            f"Original error: {e}"
        ) from e
    from .hf_utils import pick_device
    device = pick_device()
    model.to(device).eval()

    def teacher(image_path: str) -> str:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        answer = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        from .data import strip_think
        return strip_think(answer)

    return teacher


def _image_block_b64(image_path: str) -> tuple[str, str]:
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    data = base64.standard_b64encode(Path(image_path).read_bytes()).decode()
    return mime, data


def _anthropic_vision_teacher(model_name: str, instruction: str):
    """Anthropic API as vision teacher. Sends every image to the API - do not
    use for images that must stay in your environment."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Export it first, or use "
            "--images hf for a local vision model instead."
        )
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("pip install anthropic") from e

    client = anthropic.Anthropic()

    def teacher(image_path: str) -> str:
        mime, data = _image_block_b64(image_path)
        resp = client.messages.create(
            model=model_name,
            max_tokens=2048,
            messages=[{"role": "user", "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": mime, "data": data}},
                {"type": "text", "text": instruction},
            ]}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"the API declined to describe {image_path}")
        return "".join(b.text for b in resp.content if b.type == "text")

    return teacher


def _gemini_vision_teacher(model_name: str, instruction: str):
    """Google Gemini API as vision teacher. Sends every image to the API - do
    not use for images that must stay in your environment."""
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Export it first, or use "
            "--images hf for a local vision model instead."
        )
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError("pip install google-genai") from e

    client = genai.Client()

    def teacher(image_path: str) -> str:
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        part = types.Part.from_bytes(data=Path(image_path).read_bytes(),
                                     mime_type=mime)
        resp = client.models.generate_content(model=model_name,
                                              contents=[part, instruction])
        return resp.text or ""

    return teacher


def _openai_vision_teacher(model_name: str, instruction: str):
    """OpenAI API as vision teacher. Sends every image to the API - do not
    use for images that must stay in your environment."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it first, or use "
            "--images hf for a local vision model instead."
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai") from e

    client = OpenAI()

    def teacher(image_path: str) -> str:
        mime, data = _image_block_b64(image_path)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{data}"}},
                {"type": "text", "text": instruction},
            ]}],
        )
        return resp.choices[0].message.content

    return teacher
