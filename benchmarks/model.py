"""
ExpertBank — a deliberately memory-hungry routed model for the UFM benchmark.

It is a token-choice Mixture-of-Experts: an embedding, a router, a bank of N
independent expert MLPs, and a head. Per token we route to ``top_k`` experts.
The point of the benchmark is that the *total* expert parameters can far exceed
24 GB of VRAM while only a subset is needed at any moment — exactly the sparsity
UFM is designed to exploit.

Each expert is processed as its own ``nn.Module`` so the benchmark can wrap it in
``with ufm.activate(expert): ...`` (UFM mode) or leave it resident (baseline).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ExpertBank(nn.Module):
    """A routed MoE whose expert bank is the dominant memory cost."""

    def __init__(
        self,
        vocab: int = 4096,
        d_model: int = 2048,
        d_hidden: int = 8192,
        num_experts: int = 64,
        top_k: int = 2,
    ):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.embed = nn.Embedding(vocab, d_model)
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(Expert(d_model, d_hidden) for _ in range(num_experts))
        self.head = nn.Linear(d_model, vocab)

    # --- memory accounting -------------------------------------------------
    def expert_bytes(self, dtype_size: int = 4) -> int:
        """Total bytes of the expert bank (the part UFM swaps)."""
        per = self.experts[0].num_params()
        return per * self.num_experts * dtype_size

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --- forward -----------------------------------------------------------
    def forward(self, idx: torch.Tensor, expert_ctx=None) -> torch.Tensor:
        """
        idx: [B, T] int64 token ids (on the compute device).
        expert_ctx: optional callable expert_ctx(module) -> context manager.
                    In UFM mode this prefetches/activates the expert; in baseline
                    mode pass None (experts are already resident).
        Returns logits [B, T, vocab].
        """
        import contextlib

        if expert_ctx is None:
            expert_ctx = lambda _m: contextlib.nullcontext()

        x = self.embed(idx)  # [B, T, d]
        B, T, d = x.shape
        flat = x.reshape(B * T, d)  # [N, d]

        logits = self.router(flat)  # [N, E]
        weights, sel = torch.topk(logits, self.top_k, dim=-1)  # [N, k]
        weights = torch.softmax(weights, dim=-1)

        out = torch.zeros_like(flat)
        # Dispatch expert-by-expert so each expert is touched exactly once per step.
        for e in range(self.num_experts):
            # tokens (and which of their k slots) routed to expert e
            mask = sel == e  # [N, k]
            if not bool(mask.any()):
                continue
            tok_idx, slot_idx = mask.nonzero(as_tuple=True)
            expert = self.experts[e]
            with expert_ctx(expert):
                y = expert(flat[tok_idx])  # [m, d]
            w = weights[tok_idx, slot_idx].unsqueeze(-1)  # [m, 1]
            out.index_add_(0, tok_idx, y * w)

        out = out.reshape(B, T, d)
        return self.head(out)
