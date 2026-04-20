"""Geometric-structure pretraining loss."""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from ..registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("geometric_structure")
class GeometricStructureLoss:
    """Weighted sum of atom-mask CE, coord-denoise MSE, and charge MSE.

    Config example::

        loss:
          type: geometric_structure
          atom_weight: 1.0
          coord_weight: 1.0
          charge_weight: 1.0
    """

    def __init__(
        self,
        atom_weight: float = 1.0,
        coord_weight: float = 1.0,
        charge_weight: float = 1.0,
        lidi_node_weight: float = 0.0,
        lidi_edge_weight: float = 0.0,
        lidi_conservation_weight: float = 0.0,
        lidi_reduce: str = "global",
        loss_balance_mode: str = "fixed",
        loss_ema_beta: float = 0.98,
        loss_balance_eps: float = 1e-6,
        lidi_edge_transform: str = "none",
        lidi_edge_scale: float = 1e-2,
    ) -> None:
        self.atom_weight = float(atom_weight)
        self.coord_weight = float(coord_weight)
        self.charge_weight = float(charge_weight)
        self.lidi_node_weight = float(lidi_node_weight)
        self.lidi_edge_weight = float(lidi_edge_weight)
        self.lidi_conservation_weight = float(lidi_conservation_weight)
        self.lidi_reduce = str(lidi_reduce)
        self.loss_balance_mode = str(loss_balance_mode)
        self.loss_ema_beta = float(loss_ema_beta)
        self.loss_balance_eps = float(loss_balance_eps)
        self.lidi_edge_transform = str(lidi_edge_transform)
        self.lidi_edge_scale = float(lidi_edge_scale)

        if self.lidi_reduce not in {"global", "per_molecule"}:
            raise ValueError(
                f"lidi_reduce must be one of {{'global', 'per_molecule'}}, got {self.lidi_reduce!r}"
            )
        if self.loss_balance_mode not in {"fixed", "ema"}:
            raise ValueError(
                f"loss_balance_mode must be one of {{'fixed', 'ema'}}, got {self.loss_balance_mode!r}"
            )
        if not (0.0 <= self.loss_ema_beta < 1.0):
            raise ValueError(
                f"loss_ema_beta must be in [0, 1), got {self.loss_ema_beta}"
            )
        if self.loss_balance_eps <= 0.0:
            raise ValueError(
                f"loss_balance_eps must be > 0, got {self.loss_balance_eps}"
            )
        if self.lidi_edge_transform not in {"none", "asinh"}:
            raise ValueError(
                f"lidi_edge_transform must be one of {{'none', 'asinh'}}, got {self.lidi_edge_transform!r}"
            )
        if self.lidi_edge_scale <= 0.0:
            raise ValueError(
                f"lidi_edge_scale must be > 0, got {self.lidi_edge_scale}"
            )

        # EMA trackers for dynamic balancing of LIDI subtasks.
        self._ema_lidi_node: Optional[Tensor] = None
        self._ema_lidi_edge: Optional[Tensor] = None
        self._ema_lidi_conservation: Optional[Tensor] = None

    def metric_keys(self) -> Tuple[str, ...]:
        return (
            "loss",
            "atom_loss",
            "coord_loss",
            "charge_loss",
            "lidi_node_loss",
            "lidi_edge_loss",
            "lidi_conservation_loss",
        )

    def __call__(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute the geometric-structure loss.

        Expects *outputs* to contain (when available):
        - ``atom_logits``      – from atom-mask head
        - ``coords_denoised``  – from coord-denoise head
        - ``charge_pred``      – from charge head

        Returns:
            Dict with keys: ``loss``, ``atom_loss``, ``coord_loss``, ``charge_loss``.
        """
        zero = self._zero(outputs)

        atom_loss = zero
        if "atom_logits" in outputs and (
            "target_atomic_numbers" in batch or "atomic_numbers" in batch
        ):
            atom_loss = self._atom_loss(outputs, batch)

        coord_loss = zero
        if "coords_denoised" in outputs and (
            "coords_target" in batch or "coords" in batch
        ):
            coord_loss = self._coord_loss(outputs, batch)

        charge_loss = zero
        if "charge_pred" in outputs and "charges" in batch:
            charge_loss = self._charge_loss(outputs, batch)

        lidi_node_loss = zero
        lidi_edge_loss = zero
        lidi_conservation_loss = zero
        if "lidi_pred" in outputs and "lidi_matrix" in batch:
            lidi_node_loss = self._lidi_node_loss(outputs, batch)
            lidi_edge_loss = self._lidi_edge_loss(outputs, batch)
            lidi_conservation_loss = self._lidi_conservation_loss(outputs, batch)

        lidi_node_term = self._balance_lidi_loss(lidi_node_loss, key="node")
        lidi_edge_term = self._balance_lidi_loss(lidi_edge_loss, key="edge")
        lidi_conservation_term = self._balance_lidi_loss(
            lidi_conservation_loss, key="conservation"
        )

        total = (
            self.atom_weight * atom_loss
            + self.coord_weight * coord_loss
            + self.charge_weight * charge_loss
            + self.lidi_node_weight * lidi_node_term
            + self.lidi_edge_weight * lidi_edge_term
            + self.lidi_conservation_weight * lidi_conservation_term
        )
        return {
            "loss": total,
            "atom_loss": atom_loss,
            "coord_loss": coord_loss,
            "charge_loss": charge_loss,
            "lidi_node_loss": lidi_node_loss,
            "lidi_edge_loss": lidi_edge_loss,
            "lidi_conservation_loss": lidi_conservation_loss,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero(outputs: Dict[str, Tensor]) -> Tensor:
        for v in outputs.values():
            if isinstance(v, Tensor):
                return v.sum() * 0.0
        return torch.tensor(0.0)

    @staticmethod
    def _atom_loss(outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        import torch.nn.functional as F
        import warnings
        
        logits = outputs["atom_logits"]   # (B, N, vocab)
        targets = batch.get("target_atomic_numbers", batch["atomic_numbers"])  # (B, N)
        mask = batch.get("mask", batch.get("mask_positions", None))    # (B, N) bool, True = masked position
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        if mask is not None:
            mask_flat = mask.reshape(-1).float()
            num_masked = mask_flat.sum()
            
            # Validate that we have at least one masked position
            if num_masked == 0:
                warnings.warn(
                    "No masked positions found in batch for atom_loss, returning zero loss. "
                    "This may indicate a data pipeline issue.",
                    RuntimeWarning,
                    stacklevel=3
                )
                return loss.sum() * 0.0
            
            return (loss * mask_flat).sum() / num_masked
        return loss.mean()

    @staticmethod
    def _coord_loss(outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        import torch.nn.functional as F
        import warnings
        
        pred = outputs["coords_denoised"]   # (B, N, 3)
        target = batch.get("coords_target", batch["coords"])     # (B, N, 3)
        pad = batch.get("atom_padding", None)  # (B, N) True = pad
        diff = F.mse_loss(pred, target, reduction="none").mean(dim=-1)  # (B, N)
        if pad is not None:
            valid = (~pad).float()
            num_valid = valid.sum()
            
            # Validate that we have at least one valid atom
            if num_valid == 0:
                warnings.warn(
                    "No valid atoms found in batch for coord_loss, returning zero loss. "
                    "This may indicate a data pipeline issue.",
                    RuntimeWarning,
                    stacklevel=3
                )
                return diff.sum() * 0.0
            
            return (diff * valid).sum() / num_valid
        return diff.mean()

    @staticmethod
    def _charge_loss(outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        import torch.nn.functional as F
        import warnings
        
        pred = outputs["charge_pred"].squeeze(-1)  # (B, N)
        target = batch["charges"]                  # (B, N)
        pad = batch.get("atom_padding", None)
        loss = F.mse_loss(pred, target, reduction="none")  # (B, N)
        if pad is not None:
            valid = (~pad).float()
            num_valid = valid.sum()
            
            # Validate that we have at least one valid atom
            if num_valid == 0:
                warnings.warn(
                    "No valid atoms found in batch for charge_loss, returning zero loss. "
                    "This may indicate a data pipeline issue.",
                    RuntimeWarning,
                    stacklevel=3
                )
                return loss.sum() * 0.0
            
            return (loss * valid).sum() / num_valid
        return loss.mean()

    @staticmethod
    def _lidi_pair_valid_mask(outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        pred = outputs["lidi_pred"]
        bsz, n_atoms, _ = pred.shape
        valid = torch.ones((bsz, n_atoms), dtype=torch.bool, device=pred.device)

        atom_padding = batch.get("atom_padding")
        if isinstance(atom_padding, Tensor):
            valid = valid & (~atom_padding.to(device=pred.device))

        pair_valid = valid[:, :, None] & valid[:, None, :]
        lidi_valid = batch.get("lidi_valid")
        if isinstance(lidi_valid, Tensor):
            pair_valid = pair_valid & lidi_valid.to(device=pred.device, dtype=torch.bool)
        return pair_valid

    @staticmethod
    def _masked_mse(
        pred: Tensor,
        target: Tensor,
        mask: Tensor,
        reduction: str,
    ) -> Tensor:
        mask_f = mask.to(dtype=pred.dtype)
        sq = (pred - target).pow(2)
        if reduction == "global":
            denom = mask_f.sum()
            if denom <= 0:
                return pred.sum() * 0.0
            return (sq * mask_f).sum() / denom
        if reduction == "per_molecule":
            bsz = pred.shape[0]
            sq_sum = (sq * mask_f).reshape(bsz, -1).sum(dim=1)
            denom = mask_f.reshape(bsz, -1).sum(dim=1)
            valid = denom > 0
            if valid.any():
                return (sq_sum[valid] / denom[valid]).mean()
            return pred.sum() * 0.0
        raise ValueError(f"Unsupported reduction mode: {reduction}")

    def _balance_lidi_loss(self, loss: Tensor, key: str) -> Tensor:
        if self.loss_balance_mode == "fixed":
            return loss

        if key == "node":
            attr = "_ema_lidi_node"
        elif key == "edge":
            attr = "_ema_lidi_edge"
        elif key == "conservation":
            attr = "_ema_lidi_conservation"
        else:
            raise ValueError(f"Unknown LIDI balance key: {key!r}")

        current = loss.detach()
        ema = getattr(self, attr)
        if ema is None or ema.device != current.device or ema.dtype != current.dtype:
            ema = current
        else:
            ema = self.loss_ema_beta * ema + (1.0 - self.loss_ema_beta) * current
        setattr(self, attr, ema)
        return loss / ema.clamp_min(self.loss_balance_eps)

    def _lidi_node_loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:

        pred = outputs["lidi_pred"]
        target = batch["lidi_matrix"].to(device=pred.device, dtype=pred.dtype)
        pair_valid = self._lidi_pair_valid_mask(outputs, batch)
        diag_mask = torch.eye(pred.shape[1], dtype=torch.bool, device=pred.device).unsqueeze(0)
        node_mask = pair_valid & diag_mask
        return self._masked_mse(pred, target, node_mask, reduction=self.lidi_reduce)

    def _lidi_edge_loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:

        pred = outputs["lidi_pred"]
        target = batch["lidi_matrix"].to(device=pred.device, dtype=pred.dtype)
        pair_valid = self._lidi_pair_valid_mask(outputs, batch)
        diag_mask = torch.eye(pred.shape[1], dtype=torch.bool, device=pred.device).unsqueeze(0)
        edge_mask = pair_valid & (~diag_mask)

        if self.lidi_edge_transform == "asinh":
            scale = pred.new_tensor(self.lidi_edge_scale)
            pred_edge = torch.asinh(pred / scale)
            target_edge = torch.asinh(target / scale)
        else:
            pred_edge = pred
            target_edge = target
        return self._masked_mse(pred_edge, target_edge, edge_mask, reduction=self.lidi_reduce)

    def _lidi_conservation_loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        import torch.nn.functional as F

        pred = outputs["lidi_pred"]
        target = batch["lidi_matrix"].to(device=pred.device, dtype=pred.dtype)
        pair_valid = self._lidi_pair_valid_mask(outputs, batch).to(dtype=pred.dtype)
        pred_masked = pred * pair_valid
        pred_total = pred_masked.sum(dim=(1, 2))
        pred_diag = torch.diagonal(pred_masked, dim1=1, dim2=2).sum(dim=1)
        pred_electron = 0.5 * (pred_total + pred_diag)

        electron_count = batch.get("electron_count")
        if isinstance(electron_count, Tensor):
            target_electron = electron_count.to(device=pred.device, dtype=pred.dtype)
        else:
            target_masked = target * pair_valid
            target_total = target_masked.sum(dim=(1, 2))
            target_diag = torch.diagonal(target_masked, dim1=1, dim2=2).sum(dim=1)
            target_electron = 0.5 * (target_total + target_diag)
        return F.mse_loss(pred_electron, target_electron)
