"""
STRATUM - Specialized Training via Reusable Adapter Tiles and Unified Merging.

Build industry-specific small language models on commodity hardware by training
independent skill "strata" (adapters) and fusing them into one model.

Public API:
    from stratum import Muon, merge, merge_strata, train_tile, run_eval

The training and eval entry points import torch/transformers lazily, so
`import stratum` stays fast and works in tools that only need the merge math.
"""
__version__ = "0.1.0"

from .muon import Muon, newton_schulz
from .merge import merge, merge_strata, extract_deltas, load_stratum_factors

__all__ = ["Muon", "newton_schulz", "merge", "merge_strata", "extract_deltas",
           "load_stratum_factors", "train_tile", "run_eval", "distill_tile",
           "__version__"]


def __getattr__(name):
    # Heavy modules load only when asked for, keeping plain imports light.
    if name == "train_tile":
        from .train import train_tile
        return train_tile
    if name == "run_eval":
        from .evaluate import run_eval
        return run_eval
    if name == "distill_tile":
        from .distill import distill_tile
        return distill_tile
    raise AttributeError(f"module 'stratum' has no attribute '{name}'")
