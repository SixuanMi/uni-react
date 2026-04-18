"""Distributed-training helpers: process-group init/cleanup and rank utilities."""
import os
from typing import Tuple

import torch
import torch.distributed as dist


def init_distributed(requested_device: str) -> Tuple[bool, int, int, int, torch.device]:
    """Initialise the process group and return (distributed, rank, world_size, local_rank, device).

    Reads WORLD_SIZE / RANK / LOCAL_RANK from the environment (set by torchrun).
    Falls back to single-process mode when WORLD_SIZE == 1.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if requested_device == "cuda" and torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            cuda_device = torch.device("cuda", local_rank)
            try:
                dist.init_process_group(
                    backend="nccl", init_method="env://", device_id=cuda_device
                )
            except (TypeError, AttributeError):
                # Older PyTorch versions may not accept `device_id`.
                # Some releases also error when the argument type is unsupported.
                dist.init_process_group(backend="nccl", init_method="env://")
            device = cuda_device
        else:
            device = torch.device("cuda")
    else:
        if distributed:
            dist.init_process_group(backend="gloo", init_method="env://")
        device = torch.device("cpu")

    return distributed, rank, world_size, local_rank, device


def cleanup_distributed(distributed: bool) -> None:
    """Destroy the process group if it was previously initialised."""
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Return ``True`` for the rank-0 (logging/checkpoint) process."""
    return rank == 0
