"""Checkpoint save / load / restart utilities."""
import argparse
import time
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from .distributed import is_main_process


ConfigLike = Union[argparse.Namespace, Mapping[str, Any]]


def _config_to_dict(config: Optional[ConfigLike]) -> Optional[Dict[str, Any]]:
    if config is None:
        return None
    if isinstance(config, Mapping):
        return dict(config)
    return vars(config)


def build_checkpoint_dict(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    distributed: bool,
    world_size: int,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Optional[Dict[str, float]],
    step_in_epoch: Optional[int] = None,
    include_optimizer: bool = True,
    best_val: Optional[float] = None,
) -> Dict:
    """Assemble a checkpoint dict ready to be passed to :func:`torch.save`."""
    state_model = model.module.state_dict() if distributed else model.state_dict()
    payload = {
        "epoch": epoch,
        "model": state_model,
        "args": vars(args),
        "train": train_metrics,
        "val": val_metrics,
        "world_size": world_size,
        "time": time.time(),
        "best_val": None if best_val is None else float(best_val),
    }
    if step_in_epoch is not None:
        payload["step_in_epoch"] = step_in_epoch
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def load_restart_checkpoint(
    restart_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    distributed: bool,
) -> Tuple[int, int, float, Optional[Dict], bool, Optional[int]]:
    """Load a training checkpoint and return resume state.

    Returns:
        (start_epoch, step_in_epoch, best_val, ckpt_args, optimizer_loaded, ckpt_world_size)
    """
    ckpt = torch.load(restart_path, map_location=device)
    state_model = ckpt.get("model", None)
    if state_model is None:
        raise ValueError(f"Checkpoint missing 'model' key: {restart_path}")

    target_model = model.module if distributed else model
    target_model.load_state_dict(state_model)

    optimizer_loaded = False
    if "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        optimizer_loaded = True

    ckpt_epoch = int(ckpt.get("epoch", 0))
    step_in_epoch = int(ckpt.get("step_in_epoch", 0) or 0)
    if step_in_epoch < 0:
        raise ValueError(f"Invalid step_in_epoch in checkpoint: {step_in_epoch}")

    start_epoch = ckpt_epoch if step_in_epoch > 0 else ckpt_epoch + 1
    start_epoch = max(1, start_epoch)

    best_val = float("inf")
    saved_best_val = ckpt.get("best_val", None)
    if saved_best_val is not None:
        best_val = float(saved_best_val)
    else:
        val_metrics = ckpt.get("val", None)
        if (
            isinstance(val_metrics, dict)
            and "loss" in val_metrics
            and val_metrics["loss"] is not None
        ):
            best_val = float(val_metrics["loss"])

    ckpt_args = ckpt.get("args", None)
    ckpt_world_size = ckpt.get("world_size", None)
    if ckpt_world_size is not None:
        ckpt_world_size = int(ckpt_world_size)

    return start_epoch, step_in_epoch, best_val, ckpt_args, optimizer_loaded, ckpt_world_size


def validate_restart_config(
    ckpt_args: Optional[Dict],
    cur_args: ConfigLike,
    ignore_config_mismatch: bool,
    rank: int,
    step_in_epoch: int = 0,
    current_world_size: int = 1,
    ckpt_world_size: Optional[int] = None,
    logger=None,
) -> None:
    """Warn or raise when a restart checkpoint was trained with different hyper-parameters.

    *Strict* keys (architecture + data-augmentation) cause an error unless
    ``ignore_config_mismatch`` is set.  *Warn* keys (optimisation schedule) only
    print a warning.
    """
    strict_keys = [
        "train_mode", "emb_dim", "inv_layer", "se3_layer", "heads",
        "atom_vocab_size", "cutoff", "num_kernel",
        "path_dropout", "activation_dropout", "attn_dropout",
        "mask_ratio", "mask_token_id", "min_masked", "max_masked",
        "noise_std", "no_center_coords", "no_recenter_noisy",
        "geometric_subtasks",
        "lidi_matrix_key", "lidi_offsets_key", "lidi_values_key",
        "vip_vea_keys", "fukui_keys",
        "target", "targets", "split",
        "head_hidden_dim", "head_dropout",
    ]
    warn_keys = [
        "lr", "descriptor_lr", "task_lr", "weight_decay",
        "atom_weight", "coord_weight", "charge_weight",
        "lidi_node_weight", "lidi_edge_weight", "lidi_conservation_weight",
        "vip_vea_weight", "fukui_weight",
        "batch_size", "num_workers", "train_h5", "val_h5",
        "freeze_backbone_epochs", "pretrained_ckpt", "data_root",
    ]

    if not isinstance(ckpt_args, dict):
        return

    cur = _config_to_dict(cur_args)
    if cur is None:
        return
    strict_mismatch = [
        (k, ckpt_args[k], cur[k])
        for k in strict_keys
        if k in ckpt_args and k in cur and ckpt_args[k] != cur[k]
    ]
    warn_mismatch = [
        (k, ckpt_args[k], cur[k])
        for k in warn_keys
        if k in ckpt_args and k in cur and ckpt_args[k] != cur[k]
    ]

    # Mid-epoch restarts are even stricter about batch composition.
    if step_in_epoch > 0:
        for k in ("batch_size", "train_h5"):
            if k in ckpt_args and k in cur and ckpt_args[k] != cur[k]:
                if not any(x[0] == k for x in strict_mismatch):
                    strict_mismatch.append((k, ckpt_args[k], cur[k]))

    if strict_mismatch and not ignore_config_mismatch:
        lines = [
            "Restart config mismatch on strict keys. "
            "Use --restart_ignore_config to bypass.",
        ]
        lines += [f"  - {k}: ckpt={old!r}, current={new!r}" for k, old, new in strict_mismatch]
        raise ValueError("\n".join(lines))

    if strict_mismatch and is_main_process(rank):
        msg = "strict key mismatch (ignored): " + ", ".join(k for k, _, _ in strict_mismatch)
        if logger is not None:
            logger.log({"message": msg}, phase="restart_warning")
        else:
            print(f"[restart] warning: {msg}")

    if warn_mismatch and is_main_process(rank):
        if logger is not None:
            for k, old, new in warn_mismatch:
                logger.log({"key": k, "ckpt": old, "current": new}, phase="restart_warning")
        else:
            print("[restart] warning: config changed on non-strict keys:")
            for k, old, new in warn_mismatch:
                print(f"  - {k}: ckpt={old!r}, current={new!r}")


def load_init_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    strict: bool = False,
    rank: int = 0,
    logger=None,
) -> None:
    """Load a warm-start checkpoint into a model.

    Supports full checkpoints with ``model`` key, raw state_dicts, and raw
    descriptor state_dicts that need a ``descriptor.`` prefix.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    if state:
        model_keys = set(model.state_dict().keys())
        has_descriptor_prefix = any(k.startswith("descriptor.") for k in state.keys())

        # Full pretrain checkpoints store backbone weights under ``descriptor.*``.
        # When warm-starting a bare encoder (stage-3 reaction pretraining), strip
        # that prefix so the encoder can consume the weights directly.
        if has_descriptor_prefix and not any(k.startswith("descriptor.") for k in model_keys):
            remapped = {}
            for k, v in state.items():
                if k.startswith("descriptor."):
                    stripped = k[len("descriptor."):]
                    remapped[stripped if stripped in model_keys else k] = v
                else:
                    remapped[k] = v
            state = remapped
        elif not has_descriptor_prefix and hasattr(model, "descriptor"):
            desc_keys = set(model.descriptor.state_dict().keys())  # type: ignore[attr-defined]
            remapped = {}
            for k, v in state.items():
                if k in desc_keys:
                    remapped[f"descriptor.{k}"] = v
                else:
                    remapped[k] = v
            state = remapped

    ret = model.load_state_dict(state, strict=strict)
    if is_main_process(rank):
        if logger is not None:
            logger.log({"ckpt": ckpt_path, "strict": bool(strict)}, phase="init")
            if not strict:
                logger.log(
                    {"missing": len(ret.missing_keys), "unexpected": len(ret.unexpected_keys)},
                    phase="init_state",
                )
        else:
            print(f"[init] loaded {ckpt_path}")
            if not strict:
                print(f"[init] missing={len(ret.missing_keys)} unexpected={len(ret.unexpected_keys)}")
