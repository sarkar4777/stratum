"""
Recipe loading and validation for `stratum stack`.

A recipe is the reproducible build file you check into version control, so a
typo in it must fail loudly BEFORE hours of training - a misspelled `epoch:`
silently ignored would train with the default and nobody would know. Every key
is checked against a whitelist and unknown keys name their closest intent.

Recipe shape (see examples/recipe.yaml):

  base_model: Qwen/Qwen3-1.7B          # required
  output_model: models/my-slm          # required
  optimizer: muon                      # defaults for every stratum,
  system: "..."                        # overridable per stratum
  load_4bit: true
  lr: 2e-2
  batch_size: 4
  grad_accum: 4
  max_len: 1024
  seed: 42
  strata:                              # required, at least one
    - name: extract                    # required
      skill: examples/extract.jsonl    # required
      out: strata/extract              # required
      rank: 16
      epochs: 3
      ... any of the training defaults above, per stratum ...
      distill:                         # optional - makes this stratum distilled
        teacher: Qwen/Qwen3-4B         # required if distill is present
        temperature: 2.0
        alpha: 0.5
        teacher_4bit: false
  merge:
    method: linear                     # linear / ties / dare
    weights: [1.0, 1.0]
    density: 0.2                       # ties only
    drop: 0.9                          # dare only
    seed: 42                           # dare only
"""
from __future__ import annotations

from pathlib import Path

TOP_KEYS = {"base_model", "output_model", "optimizer", "system", "load_4bit",
            "lr", "adamw_lr", "batch_size", "grad_accum", "max_len", "seed",
            "strata", "merge"}
STRATUM_KEYS = {"name", "skill", "out", "rank", "epochs", "optimizer", "system",
                "load_4bit", "lr", "adamw_lr", "batch_size", "grad_accum",
                "max_len", "seed", "distill"}
DISTILL_KEYS = {"teacher", "temperature", "alpha", "teacher_4bit", "batch_size"}
MERGE_KEYS = {"method", "weights", "density", "drop", "seed"}


def _check_keys(given: dict, allowed: set, where: str):
    unknown = sorted(set(given) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown key(s) {unknown} in {where}. Allowed: {sorted(allowed)}. "
            f"A misspelled key would be silently ignored otherwise, so STRATUM "
            f"refuses to guess."
        )


def load_recipe(path: str) -> dict:
    """Read and validate a recipe YAML. Raises ValueError with the exact
    problem rather than training for hours on a half-read file."""
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Recipe not found: {path}")
    recipe = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise ValueError(f"{path} is not a YAML mapping.")

    _check_keys(recipe, TOP_KEYS, path)
    for req in ("base_model", "output_model", "strata"):
        if req not in recipe:
            raise ValueError(f"{path} is missing required key '{req}'.")
    if not isinstance(recipe["strata"], list) or not recipe["strata"]:
        raise ValueError(f"{path}: 'strata' must be a non-empty list.")

    for i, st in enumerate(recipe["strata"]):
        where = f"{path} strata[{i}]"
        if not isinstance(st, dict):
            raise ValueError(f"{where} must be a mapping.")
        _check_keys(st, STRATUM_KEYS, where)
        for req in ("name", "skill", "out"):
            if req not in st:
                raise ValueError(f"{where} is missing required key '{req}'.")
        if "distill" in st:
            _check_keys(st["distill"], DISTILL_KEYS, f"{where}.distill")
            if "teacher" not in st["distill"]:
                raise ValueError(f"{where}.distill is missing required key 'teacher'.")

    merge = recipe.get("merge", {})
    _check_keys(merge, MERGE_KEYS, f"{path} merge")
    weights = merge.get("weights")
    if weights is not None and len(weights) != len(recipe["strata"]):
        raise ValueError(
            f"{path}: merge.weights has {len(weights)} entries for "
            f"{len(recipe['strata'])} strata."
        )
    return recipe


def stratum_setting(recipe: dict, st: dict, key: str, default):
    """A per-stratum setting falls back to the recipe-wide value, then the
    built-in default - so you set batch_size once at the top and override it
    only where a stratum needs something else."""
    if key in st:
        return st[key]
    return recipe.get(key, default)
