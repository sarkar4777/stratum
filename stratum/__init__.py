"""
STRATUM - Specialized Training via Reusable Adapter Tiles and Unified Merging.

Build industry-specific small language models on commodity hardware by training
independent skill "strata" (adapters) and fusing them into one model.

Public API:
    from stratum import Muon, merge, train_tile, run_eval
"""
__version__ = "0.1.0"

from .muon import Muon, newton_schulz
from .merge import merge, extract_deltas

__all__ = ["Muon", "newton_schulz", "merge", "extract_deltas", "__version__"]
