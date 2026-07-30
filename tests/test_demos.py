"""Guards for the claims the docs make with demo numbers.

Two kinds of test. The subprocess test proves scripts/demo_concepts.py runs
clean end to end (its printed numbers are quoted verbatim in docs 3, 4, 5,
and 7, so a crash or a renamed demo would silently orphan those quotes).
The torch tests re-prove the demos' substantive claims against the real
stratum code, so the library and the teaching material can't drift apart.
"""
import subprocess
import sys
from pathlib import Path

import torch

from stratum.merge import merge

ROOT = Path(__file__).parent.parent


def test_demo_script_runs_and_covers_all_demos():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "demo_concepts.py")],
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    for n in range(1, 7):
        assert f"DEMO {n}" in result.stdout
    # The race table must include all three contenders.
    for opt in ("muon", "adamw", "sgd"):
        assert opt in result.stdout


def test_ties_beats_linear_on_crosstalk():
    """The doc 5 demo claim, re-proven against stratum.merge itself:
    when strata carry weak opposite-signed noise on each other's strong
    entries, TIES recovers the ideal merge better than linear addition."""
    torch.manual_seed(3)
    g, strong, noise = 8, 0.5, 0.12
    d1 = torch.zeros(g, g)
    d2 = torch.zeros(g, g)
    d1[:4, :] = strong
    d2[4:, :] = -strong
    d1[4:, :] = noise * torch.sign(torch.randn(4, g))
    d2[:4, :] = noise * torch.sign(torch.randn(4, g))

    ideal = torch.zeros(g, g)
    ideal[:4, :] = strong
    ideal[4:, :] = -strong

    linear = merge("linear", [{"w": d1}, {"w": d2}])["w"]
    ties = merge("ties", [{"w": d1}, {"w": d2}], density=0.5)["w"]

    err_linear = torch.norm(linear - ideal)
    err_ties = torch.norm(ties - ideal)
    assert err_ties < err_linear


def test_temperature_softens_distributions():
    """The doc 7 demo claim: higher temperature shrinks the gap between the
    top pick and the rest, without changing their order."""
    logits = torch.tensor([4.0, 2.5, 1.0, 0.5])
    p1 = torch.softmax(logits / 1.0, dim=0)
    p2 = torch.softmax(logits / 2.0, dim=0)
    p4 = torch.softmax(logits / 4.0, dim=0)
    # The winner's share falls as T rises, the runner-up's share grows.
    assert p1[0] > p2[0] > p4[0]
    assert p1[1] < p2[1] < p4[1]
    # The order never changes - softening reveals judgment, not chaos.
    for p in (p1, p2, p4):
        assert torch.equal(p.argsort(descending=True),
                           torch.tensor([0, 1, 2, 3]))
