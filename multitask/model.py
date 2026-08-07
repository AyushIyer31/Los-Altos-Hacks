"""Multi-task stability network (PyTorch).

ONE shared sequence encoder feeding FOUR separate regression heads — one per
measurement_type in stability_dataset_multitask.csv:

    ddG  · dTm · Tm · abundance

The encoder consumes precomputed ESM-2 (esm2_t30_150M_UR50D, 640-dim) mean-pooled
embeddings of the wild-type and mutant sequences, plus their difference, so each
row is represented as a stability "shift" on top of the wild-type fold. A FiLM
layer conditions that shared representation on assay (temperature, pH); rows with
no real conditions (~49% of the dataset — Domainome / Meltome) carry a learned
"no-condition" flag instead of a faked value.

This is a runnable SCAFFOLD: correct structure, sensible defaults. The obvious
upgrade path (noted inline) is per-residue / mutation-site embeddings instead of
mean-pooling, and optional end-to-end ESM fine-tuning.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Fixed head order — used everywhere to map measurement_type <-> head index.
MEASUREMENT_TYPES = ["ddG", "dTm", "Tm", "abundance"]
TYPE_TO_IDX = {t: i for i, t in enumerate(MEASUREMENT_TYPES)}

# Condition vector layout: [temperature_norm, pH_norm, has_condition_flag]
COND_DIM = 3


class FiLM(nn.Module):
    """Feature-wise linear modulation: h -> gamma(cond) * h + beta(cond).

    Initialised to the identity (gamma=1, beta=0) so the network starts as if
    unconditioned and only *learns* to use (T, pH) where it helps.
    """

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.gamma(cond) * h + self.beta(cond)


def _head(hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden, hidden // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden // 2, 1),
    )


class MultiTaskStabilityNet(nn.Module):
    """Shared encoder + FiLM(T, pH) + four measurement-type heads."""

    def __init__(
        self,
        emb_dim: int = 640,
        hidden: int = 512,
        dropout: float = 0.2,
        cond_dim: int = COND_DIM,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        in_dim = emb_dim * 3  # [wt, mut, mut - wt]

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.film = FiLM(cond_dim, hidden)
        self.heads = nn.ModuleDict({t: _head(hidden, dropout) for t in MEASUREMENT_TYPES})

    def forward(self, wt: torch.Tensor, mut: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Return predictions for ALL heads, shape [batch, n_heads].

        The training loop selects the column matching each row's measurement_type.
        For whole-protein rows (Meltome Tm, no mutant) pass mut == wt so the
        difference term is zero.
        """
        x = torch.cat([wt, mut, mut - wt], dim=-1)
        h = self.encoder(x)
        h = self.film(h, cond)
        outs = [self.heads[t](h) for t in MEASUREMENT_TYPES]  # each [batch, 1]
        return torch.cat(outs, dim=-1)  # [batch, n_heads]


if __name__ == "__main__":
    # Smoke test (no GPU / no data needed): verify shapes wire up.
    net = MultiTaskStabilityNet()
    b = 8
    wt = torch.randn(b, 640)
    mut = torch.randn(b, 640)
    cond = torch.randn(b, COND_DIM)
    out = net(wt, mut, cond)
    print("output shape:", tuple(out.shape), "(expected", (b, len(MEASUREMENT_TYPES)), ")")
    print("params:", sum(p.numel() for p in net.parameters()))
