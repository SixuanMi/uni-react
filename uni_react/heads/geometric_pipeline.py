"""Geometric-structure pretraining pipeline.

Default subtasks:
  - atom mask
  - coordinate denoise
  - charge

Optional subtask:
  - LIDI matrix prediction
"""
from typing import Dict, Iterable, Optional, Set

import torch
from torch import Tensor

from .atom_mask import AtomMaskHead
from .charge import ChargeHead
from .coord_denoise import CoordDenoiseHead
from .lidi import LidiHead


class GeometricStructureTask(torch.nn.Module):
    """Bundle of geometric-structure pretraining heads.

    Composes AtomMaskHead + CoordDenoiseHead + ChargeHead and exposes a
    unified ``forward`` and ``compute_loss_dict`` interface used by
    :class:`~uni_react.model.pretrain_net.SingleMolPretrainNet`.
    """

    name = "geometric_structure"
    default_subtasks = ("atom_mask", "coord_denoise", "charge")

    def __init__(
        self,
        emb_dim: int,
        atom_vocab_size: int,
        enable_lidi: bool = False,
        lidi_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.atom_mask    = AtomMaskHead(emb_dim=emb_dim, atom_vocab_size=atom_vocab_size)
        self.coord_denoise = CoordDenoiseHead(emb_dim=emb_dim)
        self.charge        = ChargeHead(emb_dim=emb_dim)
        self.lidi = LidiHead(emb_dim=emb_dim, hidden_dim=lidi_hidden_dim) if enable_lidi else None

    @staticmethod
    def _normalize_subtasks(
        subtasks: Optional[Iterable[str]], default: Iterable[str]
    ) -> Set[str]:
        return set(default if subtasks is None else subtasks)

    def forward(
        self,
        descriptors: Dict[str, Tensor],
        active_subtasks: Optional[Iterable[str]] = None,
    ) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        st = self._normalize_subtasks(active_subtasks, self.default_subtasks)
        if "atom_mask"    in st: out.update(self.atom_mask(descriptors))
        if "coord_denoise" in st: out.update(self.coord_denoise(descriptors))
        if "charge"        in st: out.update(self.charge(descriptors))
        if "lidi" in st:
            if self.lidi is None:
                raise RuntimeError(
                    "Subtask 'lidi' requested but LIDI head is not enabled in GeometricStructureTask."
                )
            out.update(self.lidi(descriptors))
        return out

    @staticmethod
    def _zero(outputs: Dict[str, Tensor]) -> Tensor:
        for v in outputs.values():
            if isinstance(v, Tensor):
                return v.sum() * 0.0
        return torch.tensor(0.0)

    def compute_loss_dict(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        atom_weight: float = 1.0,
        coord_weight: float = 1.0,
        charge_weight: float = 1.0,
        active_subtasks: Optional[Iterable[str]] = None,
    ) -> Dict[str, Tensor]:
        st   = self._normalize_subtasks(active_subtasks, self.default_subtasks)
        zero = self._zero(outputs)

        atom_loss  = self.atom_mask.compute_loss(outputs, batch)   if "atom_mask"    in st and "atom_logits"     in outputs else zero
        coord_loss = self.coord_denoise.compute_loss(outputs, batch) if "coord_denoise" in st and "coords_denoised" in outputs else zero
        charge_loss = self.charge.compute_loss(outputs, batch)      if "charge"        in st and "charge_pred"     in outputs else zero

        total = atom_weight * atom_loss + coord_weight * coord_loss + charge_weight * charge_loss
        return {"loss": total, "atom_loss": atom_loss, "coord_loss": coord_loss, "charge_loss": charge_loss}

    def compute_loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        return self.compute_loss_dict(outputs, batch)["loss"]
