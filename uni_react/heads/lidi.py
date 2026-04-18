"""LIDI matrix prediction head."""
from typing import Dict

import torch
from torch import Tensor

from ..registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("lidi")
class LidiHead(torch.nn.Module):
    """Predicts per-molecule dense LIDI matrices from node embeddings.

    Output:
      ``lidi_pred`` with shape ``(B, N, N)``.
    """

    name = "lidi"

    def __init__(self, emb_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        pair_dim = emb_dim * 4
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(pair_dim),
            torch.nn.Linear(pair_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, descriptors: Dict[str, Tensor]) -> Dict[str, Tensor]:
        node_feats = descriptors["node_feats"]    # (B, N, D)
        atom_padding = descriptors["atom_padding"]  # (B, N)

        bsz, num_atoms, _ = node_feats.shape
        h_i = node_feats[:, :, None, :].expand(bsz, num_atoms, num_atoms, -1)
        h_j = node_feats[:, None, :, :].expand(bsz, num_atoms, num_atoms, -1)
        pair_feat = torch.cat([h_i, h_j, torch.abs(h_i - h_j), h_i * h_j], dim=-1)

        pred = self.head(pair_feat).squeeze(-1)  # (B, N, N)
        pad_2d = atom_padding[:, :, None] | atom_padding[:, None, :]
        pred = pred.masked_fill(pad_2d, 0)
        return {"lidi_pred": pred}
