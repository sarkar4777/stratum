"""
Adapter (stratum) merging.

An adapter's effect is an additive low-rank update delta = scaling * (B @ A) on
a frozen weight. Because every stratum modifies the SAME frozen base, combining
them is (weighted) addition of their deltas. This module provides:

  extract_deltas(dir) -> {base_weight_name: delta_tensor} for one stratum
  merge(method, ...) -> combined {base_weight_name: delta_tensor}

Methods (see docs/05-merging.md):
  linear : weighted sum. Default. Best when skills are fairly independent.
  ties : trim to top-magnitude entries, resolve sign conflicts by vote.
  dare : randomly drop redundant entries and rescale, reducing crosstalk.

All strata being merged MUST share the same base model. The CLI verifies this
from each stratum's stratum_card.json before calling merge().
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def extract_deltas(stratum_dir: str) -> dict[str, torch.Tensor]:
    """Reconstruct each LoRA module's full-size weight delta from a saved stratum.

    PEFT stores adapters as lora_A / lora_B tensor pairs per target module.
    The applied update is scaling * (B @ A), where scaling = lora_alpha / r.
    We key each delta by the BASE model's weight name so merges line up across
    strata and can be added directly onto base weights.
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
        raw = torch.load(bin_file, map_location="cpu")

    cfg = json.loads((path / "adapter_config.json").read_text())
    r = cfg.get("r", 16)
    alpha = cfg.get("lora_alpha", r)
    use_rslora = cfg.get("use_rslora", False)
    # rsLoRA scales by alpha/sqrt(r); classic LoRA by alpha/r.
    scaling = (alpha / (r ** 0.5)) if use_rslora else (alpha / r)

    # Find every module that has a lora_A (its lora_B pairs by shared prefix).
    prefixes = sorted({k.split(".lora_A.")[0] for k in raw if ".lora_A." in k})

    deltas: dict[str, torch.Tensor] = {}
    for pfx in prefixes:
        a_key = next((k for k in raw if k.startswith(pfx + ".lora_A.")), None)
        b_key = next((k for k in raw if k.startswith(pfx + ".lora_B.")), None)
        if a_key is None or b_key is None:
            continue
        A = raw[a_key].float() # [r, in]
        B = raw[b_key].float() # [out, r]
        delta = (B @ A) * scaling # [out, in], same shape as base weight

        # Map PEFT module name -> base weight name.
        base_name = pfx.replace("base_model.model.", "")
        if not base_name.endswith(".weight"):
            base_name += ".weight"
        deltas[base_name] = delta
    if not deltas:
        raise ValueError(f"No LoRA A/B pairs found in {stratum_dir}.")
    return deltas


def _trim(t: torch.Tensor, density: float) -> torch.Tensor:
    """Keep only the top `density` fraction of entries by absolute value."""
    if density >= 1.0:
        return t
    flat = t.abs().flatten()
    k = max(1, int(flat.numel() * density))
    thresh = torch.topk(flat, k, largest=True).values.min()
    return t * (t.abs() >= thresh)


def merge_linear(deltas_per_stratum, weights):
    """Weighted sum: combined = sum_i weight_i * delta_i."""
    keys = set().union(*[d.keys() for d in deltas_per_stratum])
    merged = {}
    for key in keys:
        acc = None
        for d, w in zip(deltas_per_stratum, weights):
            if key not in d:
                continue
            term = d[key] * w
            acc = term if acc is None else acc + term
        merged[key] = acc
    return merged


def merge_ties(deltas_per_stratum, weights, density: float = 0.2):
    """TIES: trim each stratum, elect a per-element sign, average agreeing strata."""
    keys = set().union(*[d.keys() for d in deltas_per_stratum])
    merged = {}
    for key in keys:
        tensors = []
        for d, w in zip(deltas_per_stratum, weights):
            if key in d:
                tensors.append(_trim(d[key], density) * w)
        if not tensors:
            continue
        stacked = torch.stack(tensors) # [n, out, in]
        sign = torch.sign(stacked.sum(dim=0))
        agree = (torch.sign(stacked) == sign.unsqueeze(0)) & (stacked != 0)
        kept = stacked * agree
        count = agree.sum(dim=0).clamp(min=1)
        merged[key] = kept.sum(dim=0) / count
    return merged


def merge_dare(deltas_per_stratum, weights, drop: float = 0.9):
    """DARE: randomly drop `drop` fraction of each stratum's entries, rescale, sum."""
    if not 0.0 <= drop < 1.0:
        raise ValueError(f"drop must be in [0,1): {drop}")
    processed = []
    for d in deltas_per_stratum:
        new = {}
        for key, delta in d.items():
            mask = (torch.rand_like(delta) > drop).float()
            new[key] = delta * mask / (1.0 - drop)
        processed.append(new)
    return merge_linear(processed, weights)


MERGE_METHODS = {"linear": merge_linear, "ties": merge_ties, "dare": merge_dare}


def merge(method: str, deltas_per_stratum, weights=None, **kwargs):
    """Dispatch to a merge method. weights default to equal (they need not sum to 1)."""
    if not deltas_per_stratum:
        raise ValueError("No strata to merge.")
    if weights is None:
        weights = [1.0] * len(deltas_per_stratum)
    if len(weights) != len(deltas_per_stratum):
        raise ValueError("Number of weights must match number of strata.")
    if method not in MERGE_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(MERGE_METHODS)}.")
    return MERGE_METHODS[method](deltas_per_stratum, weights, **kwargs)
