"""
Adapter (stratum) merging.

An adapter's effect is an additive low-rank update delta = scaling * (B @ A) on
a frozen weight. Because every stratum modifies the SAME frozen base, combining
them is (weighted) addition of their deltas. This module provides:

  load_stratum_factors(dir) -> {base_weight_name: (A, B, scaling)} for one stratum
  extract_deltas(dir)       -> {base_weight_name: dense_delta} (small models/tests)
  merge(method, ...)        -> combined {base_weight_name: delta_tensor}
  merge_strata(...)         -> the full pipeline: verify, merge, apply, save

Methods (see docs/05-merging.md):
  linear : weighted sum. Default. Best when skills are fairly independent.
  ties   : trim to top-magnitude entries, resolve sign conflicts by vote.
  dare   : randomly drop redundant entries and rescale, reducing crosstalk.

All strata being merged MUST share the same base model. merge_strata verifies
this from each stratum's stratum_card.json before touching any weights.

Memory note: a dense delta is the full size of its base weight, so holding
every delta for every stratum at once is several full model copies. merge_strata
avoids that by keeping only the small low-rank factors and materializing dense
deltas one weight at a time.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def load_stratum_factors(stratum_dir: str) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    """Load a stratum's LoRA factors keyed by BASE model weight name.

    PEFT stores adapters as lora_A / lora_B tensor pairs per target module.
    The applied update is scaling * (B @ A), where scaling = lora_alpha / r.
    We return (A, B, scaling) per weight - tiny compared to the dense delta -
    and let callers materialize B @ A only when and where they need it.
    """
    path = Path(stratum_dir)

    # Load adapter tensors (safetensors preferred, .bin fallback).
    st_file = path / "adapter_model.safetensors"
    if st_file.exists():
        from safetensors.torch import load_file
        raw = load_file(str(st_file))
    else:
        bin_file = path / "adapter_model.bin"
        if not bin_file.exists():
            raise FileNotFoundError(
                f"No adapter weights in {stratum_dir} "
                f"(expected adapter_model.safetensors or .bin)."
            )
        # weights_only guards against pickled code execution - strata may come
        # from other teams or other organizations.
        raw = torch.load(bin_file, map_location="cpu", weights_only=True)

    cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("use_dora"):
        raise ValueError(
            f"{stratum_dir} was trained with DoRA, which is not a plain additive "
            f"update - merging it as one would be silently wrong. Retrain the "
            f"stratum without DoRA, or keep it as a separate runtime adapter."
        )
    extra = sorted({k for k in raw if ".modules_to_save." in k})
    if extra:
        raise ValueError(
            f"{stratum_dir} contains fully-retrained modules (modules_to_save: "
            f"{extra[:3]}...). Those are weight replacements, not additive deltas, "
            f"and cannot be merged this way."
        )

    r = cfg.get("r", 16)
    alpha = cfg.get("lora_alpha", r)
    use_rslora = cfg.get("use_rslora", False)
    # rsLoRA scales by alpha/sqrt(r), classic LoRA by alpha/r.
    scaling = (alpha / (r ** 0.5)) if use_rslora else (alpha / r)

    # Find every module that has a lora_A (its lora_B pairs by shared prefix).
    prefixes = sorted({k.split(".lora_A.")[0] for k in raw if ".lora_A." in k})

    factors: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for pfx in prefixes:
        a_key = next((k for k in raw if k.startswith(pfx + ".lora_A.")), None)
        b_key = next((k for k in raw if k.startswith(pfx + ".lora_B.")), None)
        if a_key is None or b_key is None:
            continue
        A = raw[a_key].float()  # [r, in]
        B = raw[b_key].float()  # [out, r]

        # Map PEFT module name -> base weight name.
        base_name = pfx.replace("base_model.model.", "")
        if not base_name.endswith(".weight"):
            base_name += ".weight"
        factors[base_name] = (A, B, scaling)
    if not factors:
        raise ValueError(f"No LoRA A/B pairs found in {stratum_dir}.")
    return factors


def _dense(factor: tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    A, B, scaling = factor
    return (B @ A) * scaling  # [out, in], same shape as the base weight


def extract_deltas(stratum_dir: str) -> dict[str, torch.Tensor]:
    """Reconstruct every dense weight delta for one stratum.

    Convenient for small models and tests. For real merges prefer
    merge_strata(), which never holds more than one weight's deltas at a time.
    """
    return {name: _dense(f) for name, f in load_stratum_factors(stratum_dir).items()}


def _trim(t: torch.Tensor, density: float) -> torch.Tensor:
    """Keep only the top `density` fraction of entries by absolute value."""
    if density >= 1.0:
        return t
    flat = t.abs().flatten()
    k = max(1, int(flat.numel() * density))
    thresh = torch.topk(flat, k, largest=True).values.min()
    return t * (t.abs() >= thresh)


def merge_key(method: str, tensors: list[torch.Tensor], weights: list[float],
              density: float = 0.2, drop: float = 0.9,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """Merge the dense deltas for ONE weight across strata. The per-key core
    every method reduces to. merge_strata calls this one weight at a time."""
    if method == "linear":
        acc = None
        for t, w in zip(tensors, weights):
            term = t * w
            acc = term if acc is None else acc + term
        return acc

    if method == "ties":
        stacked = torch.stack([_trim(t, density) * w
                               for t, w in zip(tensors, weights)])  # [n, out, in]
        sign = torch.sign(stacked.sum(dim=0))
        agree = (torch.sign(stacked) == sign.unsqueeze(0)) & (stacked != 0)
        kept = stacked * agree
        count = agree.sum(dim=0).clamp(min=1)
        return kept.sum(dim=0) / count

    if method == "dare":
        if not 0.0 <= drop < 1.0:
            raise ValueError(f"drop must be in [0,1): {drop}")
        acc = None
        for t, w in zip(tensors, weights):
            mask = (torch.rand(t.shape, generator=generator,
                               device=t.device) > drop).float()
            term = (t * mask / (1.0 - drop)) * w
            acc = term if acc is None else acc + term
        return acc

    raise ValueError(f"Unknown method '{method}'. Choose from {list(MERGE_METHODS)}.")


MERGE_METHODS = ("linear", "ties", "dare")


def merge(method: str, deltas_per_stratum, weights=None, density: float = 0.2,
          drop: float = 0.9, seed: int | None = None):
    """Merge dicts of dense deltas. weights default to equal (they need not sum to 1).

    seed makes DARE's random dropping reproducible. linear and ties are
    deterministic anyway.
    """
    if not deltas_per_stratum:
        raise ValueError("No strata to merge.")
    if weights is None:
        weights = [1.0] * len(deltas_per_stratum)
    if len(weights) != len(deltas_per_stratum):
        raise ValueError("Number of weights must match number of strata.")
    if method not in MERGE_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(MERGE_METHODS)}.")

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    keys = set().union(*[d.keys() for d in deltas_per_stratum])
    merged = {}
    for key in sorted(keys):
        tensors, w = [], []
        for d, wt in zip(deltas_per_stratum, weights):
            if key in d:
                tensors.append(d[key])
                w.append(wt)
        merged[key] = merge_key(method, tensors, w, density=density, drop=drop,
                                generator=generator)
    return merged


def read_stratum_card(stratum_dir: str) -> dict:
    """Read and return a stratum's provenance card, or raise a clear error."""
    card_path = Path(stratum_dir) / "stratum_card.json"
    if not card_path.exists():
        raise FileNotFoundError(
            f"{stratum_dir} has no stratum_card.json - is it a STRATUM stratum?"
        )
    return json.loads(card_path.read_text(encoding="utf-8"))


def merge_strata(strata_dirs: list[str], out_dir: str, method: str = "linear",
                 weights: list[float] | None = None, density: float = 0.2,
                 drop: float = 0.9, seed: int = 42) -> dict:
    """The full merge pipeline: verify, combine, apply onto a fresh base, save.

    Verifies every stratum shares one base model, then materializes and merges
    deltas ONE WEIGHT AT A TIME, so peak memory is the base model plus a single
    layer's deltas - not a full model copy per stratum. Returns a summary dict
    (also written to <out_dir>/stratum_merge.json).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cards = [read_stratum_card(s) for s in strata_dirs]
    bases = {c["base_model"] for c in cards}
    if len(bases) != 1:
        raise ValueError(f"Strata have different base models: {bases}. "
                         f"Only strata from the same base can be merged.")
    base_model = bases.pop()

    if weights is None:
        weights = [1.0] * len(strata_dirs)
    if len(weights) != len(strata_dirs):
        raise ValueError("Number of --weights must match number of strata.")
    if method not in MERGE_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(MERGE_METHODS)}.")

    print(f"Merging {len(strata_dirs)} strata from {base_model}")
    print(f" method={method} weights={weights}")
    if any(c.get("load_4bit") for c in cards):
        print(" note: some strata were trained against a 4-bit base (QLoRA) but are\n"
              " merged onto the full-precision base. The small mismatch usually\n"
              " costs little. If eval drops, retrain those strata with --no-4bit.\n"
              " See docs/05-merging.md.")

    factors = [load_stratum_factors(s) for s in strata_dirs]

    generator = torch.Generator().manual_seed(seed) if method == "dare" else None

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    sd = model.state_dict()

    all_keys = sorted(set().union(*[f.keys() for f in factors]))
    applied, missing = 0, 0
    for key in all_keys:
        if key not in sd:
            missing += 1
            continue
        tensors, w = [], []
        for f, wt in zip(factors, weights):
            if key in f:
                tensors.append(_dense(f[key]))
                w.append(wt)
        delta = merge_key(method, tensors, w, density=density, drop=drop,
                          generator=generator)
        sd[key].add_(delta.to(sd[key].dtype))  # in place on the live weights
        applied += 1

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(outp))
    tokenizer.save_pretrained(str(outp))
    from datetime import datetime, timezone
    summary = {
        "base_model": base_model, "method": method, "weights": weights,
        "strata": [c["stratum_name"] for c in cards],
        "strata_cards": cards,  # the full provenance of every input, in one place
        "deltas_applied": applied, "deltas_unmatched": missing,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if method == "ties":
        summary["density"] = density
    if method == "dare":
        summary["drop"] = drop
        summary["seed"] = seed
    (outp / "stratum_merge.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    print(f"Applied {applied} weight deltas ({missing} unmatched). Model -> {out_dir}")
    return summary
