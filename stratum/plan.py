"""
Build planning: can THIS machine run THAT recipe, and what to do if not.

`stratum doctor` tells you about your hardware in general. This module answers
the specific question that matters before a build: for each stratum in a
recipe, roughly how much GPU memory will training need, does it fit here, and
if not - which settings would make it fit, or should the training burst run on
rented hardware instead. `stratum plan recipe.yaml` prints the answer, and
`--emit-remote` writes a script that runs the identical build and tests on any
hourly GPU service.

The estimate uses the same arithmetic as docs/01-the-memory-problem.md:

  weights    - the frozen base: ~2.2 bytes/param in bf16, ~0.7 with 4-bit
               (quantized weights plus quantization bookkeeping)
  overhead   - the LoRA adapter with its gradients and optimizer state, CUDA
               context, and workspace: small but never zero
  activations- grow with batch_size x max_len, tamed by gradient checkpointing

These are estimates, good to maybe 30 percent either way - enough to sort
"fits comfortably" from "will not fit", which is the decision that matters.
The verdict never pretends to more precision than that.
"""
from __future__ import annotations

import re
from pathlib import Path

# A verdict is "fits", "tight", or "no_fit". Tight means within the margin of
# error of the estimate - it will probably run, possibly after a retry with a
# smaller batch.
FITS, TIGHT, NO_FIT = "fits", "tight", "no_fit"


def probe_hardware() -> dict:
    """What this machine offers for training. Returns a plain dict so the
    planning logic can be tested with made-up hardware."""
    import torch

    hw = {"gpu": None, "vram_gb": 0.0, "cuda": False, "bnb": False}
    if torch.cuda.is_available():
        hw["cuda"] = True
        hw["gpu"] = torch.cuda.get_device_name(0)
        hw["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        try:
            import bitsandbytes  # noqa
            hw["bnb"] = True
        except Exception:
            hw["bnb"] = False
    return hw


def model_params_b(model_id: str) -> float | None:
    """Read a model's size in billions from its name, e.g. 'Qwen/Qwen3-1.7B'
    gives 1.7. Returns None when the name carries no size (a local folder,
    say) - callers skip estimation rather than guess."""
    name = model_id.rstrip("/\\").split("/")[-1].split("\\")[-1]
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z0-9])", name)
    return float(m.group(1)) if m else None


def estimate_vram_gb(params_b: float, load_4bit: bool = True,
                     batch_size: int = 4, max_len: int = 1024,
                     teacher_params_b: float | None = None,
                     teacher_4bit: bool = False) -> float:
    """Rough VRAM need in GB for training one stratum. See the module
    docstring for what each term is."""
    weights = params_b * (0.7 if load_4bit else 2.2)
    overhead = max(0.6, params_b * 0.3)
    activations = 0.0004 * batch_size * max_len * (params_b ** 0.5)
    total = weights + overhead + activations
    if teacher_params_b:
        # Logit distillation holds the frozen teacher too.
        total += teacher_params_b * (0.7 if teacher_4bit else 2.2)
    return total


def _verdict(need_gb: float, vram_gb: float) -> str:
    if need_gb <= vram_gb * 0.85:
        return FITS
    if need_gb <= vram_gb * 1.3:
        return TIGHT
    return NO_FIT


def plan_stratum(name: str, params_b: float | None, hw: dict,
                 load_4bit: bool, batch_size: int, max_len: int,
                 teacher_params_b: float | None = None,
                 teacher_4bit: bool = False) -> dict:
    """Plan one stratum. Returns {name, need_gb, verdict, suggestions}."""
    out = {"name": name, "need_gb": None, "verdict": None, "suggestions": []}
    if params_b is None:
        out["verdict"] = FITS
        out["suggestions"].append(
            "base size unknown (local path or unsized name) - no estimate, "
            "assuming you know it fits")
        return out

    if not hw["cuda"]:
        # No hard memory wall on CPU, only patience. Small models are workable,
        # anything bigger belongs on rented hardware.
        out["verdict"] = FITS if params_b <= 1.0 else NO_FIT
        if params_b <= 1.0:
            out["suggestions"].append("CPU only - workable at this size, but slow")
        else:
            out["suggestions"].append(
                f"CPU only - a {params_b}B base is impractical to train here")
        return out

    use_4bit = load_4bit and hw["bnb"]
    if load_4bit and not hw["bnb"]:
        out["suggestions"].append(
            "bitsandbytes is not installed, so 4-bit is off - "
            "`pip install bitsandbytes` roughly halves the need below")

    need = estimate_vram_gb(params_b, use_4bit, batch_size, max_len,
                            teacher_params_b, teacher_4bit)
    out["need_gb"] = need
    out["verdict"] = _verdict(need, hw["vram_gb"])

    if out["verdict"] is not FITS:
        # Suggest the levers in the order docs/13 recommends pulling them.
        if batch_size > 1:
            smaller = estimate_vram_gb(params_b, use_4bit, 1, max_len,
                                       teacher_params_b, teacher_4bit)
            if _verdict(smaller, hw["vram_gb"]) != NO_FIT:
                out["suggestions"].append(
                    f"batch_size 1 with grad_accum {batch_size * 4} needs "
                    f"~{smaller:.1f} GB and keeps the same effective batch")
        if not use_4bit and hw["bnb"]:
            quant = estimate_vram_gb(params_b, True, batch_size, max_len,
                                     teacher_params_b, teacher_4bit)
            out["suggestions"].append(
                f"4-bit (the default) needs ~{quant:.1f} GB")
        if teacher_params_b and not teacher_4bit:
            t4 = estimate_vram_gb(params_b, use_4bit, batch_size, max_len,
                                  teacher_params_b, True)
            out["suggestions"].append(
                f"teacher_4bit: true needs ~{t4:.1f} GB")
    return out


def plan_recipe(recipe: dict, hw: dict) -> dict:
    """Plan every stratum in a validated recipe against this hardware.

    Returns {"strata": [per-stratum plans], "verdict": worst case,
    "base_params_b": size}. The overall verdict is the worst stratum's,
    because a build is only as runnable as its hungriest step.
    """
    from .recipe import stratum_setting

    base = recipe["base_model"]
    params_b = model_params_b(base)
    plans = []
    for st in recipe["strata"]:
        dcfg = st.get("distill") or {}
        teacher_b = model_params_b(dcfg["teacher"]) if dcfg else None
        plans.append(plan_stratum(
            name=st["name"], params_b=params_b, hw=hw,
            load_4bit=stratum_setting(recipe, st, "load_4bit", True),
            batch_size=dcfg.get("batch_size",
                                stratum_setting(recipe, st, "batch_size", 4)),
            max_len=stratum_setting(recipe, st, "max_len", 1024),
            teacher_params_b=teacher_b,
            teacher_4bit=dcfg.get("teacher_4bit", False),
        ))

    order = {FITS: 0, TIGHT: 1, NO_FIT: 2}
    worst = max((p["verdict"] for p in plans), key=order.get, default=FITS)
    return {"strata": plans, "verdict": worst, "base_params_b": params_b}


def rental_advice(params_b: float | None) -> str:
    """Which rented hardware covers the build. Named services are examples,
    not endorsements - any hourly GPU rental or your own cloud tenancy works,
    which is the point: the burst is short and the build is one script."""
    if params_b is None or params_b <= 4:
        card = "a single 24 GB card (RTX 4090 / L4 class)"
    elif params_b <= 8:
        card = "a single 48 GB card (L40S / A6000 class)"
    else:
        card = "a single 80 GB card (A100 / H100 class)"
    return (f"rent {card} by the hour - RunPod, Lambda Cloud, Vast.ai, a "
            f"Colab GPU runtime, or your company's cloud all work. A full "
            f"build is typically minutes to a few hours of billing.")


def print_plan(plan: dict, hw: dict, recipe_path: str) -> None:
    """Human-readable plan report."""
    print("STRATUM plan\n" + "-" * 42)
    if hw["cuda"]:
        quant = "4-bit available" if hw["bnb"] else "no bitsandbytes"
        print(f"Hardware: {hw['gpu']}, {hw['vram_gb']:.1f} GB VRAM, {quant}")
    else:
        print("Hardware: no CUDA GPU (CPU only)")
    print(f"Recipe:   {recipe_path}\n")

    for p in plan["strata"]:
        need = f"~{p['need_gb']:.1f} GB" if p["need_gb"] else "unknown"
        label = {FITS: "fits", TIGHT: "tight", NO_FIT: "does NOT fit"}[p["verdict"]]
        print(f"  {p['name']:<12} needs {need:<10} {label}")
        for s in p["suggestions"]:
            print(f"      - {s}")
    print()

    if plan["verdict"] == FITS:
        print("Verdict: build locally. `stratum stack " + recipe_path + "`")
    elif plan["verdict"] == TIGHT:
        print("Verdict: probably fits, within the estimate's margin. Try it - "
              "and if it runs out of memory, apply the suggestions above or "
              "go remote:")
        print("  " + rental_advice(plan["base_params_b"]))
    else:
        print("Verdict: this machine cannot run the build as configured.")
        print("  " + rental_advice(plan["base_params_b"]))
        print(f"  `stratum plan {recipe_path} --emit-remote remote/` writes "
              f"the build-and-test script to run there.")
    print("\nEstimates are rough (about 30 percent either way) - they sort "
          "'fits' from 'will not fit', no more.")


REMOTE_SCRIPT = """\
#!/usr/bin/env bash
# Build and test this STRATUM recipe on a rented GPU box.
#
# On the remote machine:
#   1. copy the project there (git clone, or rsync this folder)
#   2. from the project root, run this script with bash
#   3. copy {output_tar} back and serve it anywhere
#
# The recipe carries its own eval gates, so a finished run means a TESTED
# model - if a gate fails, this script exits non-zero and packages nothing.
set -euo pipefail

pip install -e . bitsandbytes

stratum doctor
stratum plan {recipe}

stratum stack {recipe}

tar -czf {output_tar} {output_model}
echo "Done. Copy {output_tar} back to your machine."
"""


def write_remote_bundle(out_dir: str, recipe_path: str, recipe: dict) -> str:
    """Write the remote build-and-test script. Returns the script path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_model = recipe["output_model"]
    tar_name = Path(output_model).name + ".tar.gz"
    script = out / "build-and-test.sh"
    script.write_text(REMOTE_SCRIPT.format(
        recipe=recipe_path, output_model=output_model, output_tar=tar_name,
    ), encoding="utf-8", newline="\n")
    if not recipe.get("evals"):
        print("Note: this recipe has no evals section, so the remote run "
              "builds but does not test. Add eval gates to the recipe first - "
              "docs/08-evaluation.md shows how.")
    print(f"Remote build script written to {script}")
    print("Copy the whole project folder to the rented machine and run it there.")
    return str(script)
