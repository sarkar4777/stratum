"""
Muon optimizer - MomentUm Orthogonalized by Newton-schulz.

Muon trains 2D weight matrices by orthogonalizing the momentum before each step,
so every direction of the update gets a fair share instead of a few dominating.
It keeps one state tensor per parameter (momentum), versus AdamW's two, and
typically reaches good quality in fewer steps.

Scope: Muon applies ONLY to 2D matrices. The Trainer routes non-2D parameters
(embeddings, biases, layernorm gains, and the LoRA input projections in some
setups) to a small AdamW. This module provides Muon; stratum/train.py wires the
split via `split_params_for_muon`.

References: Keller Jordan's Muon; Bernstein et al. on orthogonalized updates.
This implementation favors clarity and numerical safety for laptop-scale
adapter training.
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalize a 2D matrix G (push singular values toward 1).

    Uses the quintic Newton-Schulz iteration. We deliberately never compute the
    SVD (too slow); this achieves the same flattening with a few matmuls.

    Runs the iteration in bfloat16 for speed but is numerically safe because we
    normalize G first. Returns a tensor in G's original dtype.
    """
    if G.ndim != 2:
        raise ValueError(f"newton_schulz expects a 2D matrix, got shape {tuple(G.shape)}")

    a, b, c = 3.4445, -4.7750, 2.0315
    orig_dtype = G.dtype

    X = G.to(torch.bfloat16)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    norm = X.norm()
    if norm < eps: # zero gradient - nothing to orthogonalize
        return G.clone()
    X = X / (norm + eps)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(orig_dtype)


class Muon(Optimizer):
    """Muon optimizer for 2D weight matrices.

    Args:
        params: iterable of 2D parameters only (Trainer filters these).
        lr: learning rate. Muon tolerates larger LRs than AdamW; 2e-2 is typical.
        momentum: momentum coefficient (the single state Muon keeps).
        nesterov: use Nesterov-style lookahead (recommended).
        ns_steps: Newton-Schulz iterations per step (5 is standard).
        weight_decay: decoupled weight decay (the "W" idea from AdamW).
    """

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0,1): {momentum}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")
                if g.ndim != 2:
                    raise ValueError(
                        f"Muon received a {g.ndim}D parameter of shape "
                        f"{tuple(g.shape)}. Route non-2D params to AdamW via "
                        f"split_params_for_muon()."
                    )

                # Guard: a non-finite gradient (from a bad batch or fp underflow)
                # must never enter the momentum buffer, or it poisons all future
                # steps. Skip this parameter's update this step instead.
                if not torch.isfinite(g).all():
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf

                update = newton_schulz(update, steps=ns_steps)

                # Guard: if orthogonalization produced anything non-finite, skip.
                if not torch.isfinite(update).all():
                    continue

                # Shape-aware scale so the effective step is consistent across
                # matrices of different aspect ratios (standard Muon rescaling).
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr * scale)

        return loss


def split_params_for_muon(model):
    """Split a model's *trainable* params into (matrices for Muon, rest for AdamW).

    Only 2D parameters go to Muon. Everything else (embeddings viewed as 2D are
    excluded by name, plus biases/norms) goes to AdamW. Returns two lists.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.ndim == 2
        # Embeddings and the LM head are 2D but should use AdamW - they behave
        # like lookup tables, not transform matrices.
        looks_like_embedding = any(k in name.lower() for k in ("embed", "lm_head", "wte", "wpe"))
        if is_matrix and not looks_like_embedding:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params
