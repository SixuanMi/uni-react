"""Unit tests for LIDI dataset loading/collation."""
from pathlib import Path

import h5py
import numpy as np
import torch

from uni_react.utils.dataset import H5SingleMolPretrainDataset, collate_fn_pretrain


def _write_tiny_lidi_h5(path: Path) -> None:
    # Two molecules: N=2 and N=3
    mol_offsets = np.asarray([0, 2, 5], dtype=np.int64)
    atom_numbers = np.asarray([1, 6, 1, 7, 8], dtype=np.int64)
    coords = np.zeros((5, 3), dtype=np.float32)
    charges = np.zeros((5,), dtype=np.float32)

    lidi_m0 = np.asarray([[1.0, 0.2], [0.2, 1.5]], dtype=np.float32)
    lidi_m1 = np.asarray(
        [[1.1, 0.1, 0.0], [0.1, 1.6, 0.3], [0.0, 0.3, 2.2]],
        dtype=np.float32,
    )
    lidi_flat = np.concatenate([lidi_m0.reshape(-1), lidi_m1.reshape(-1)], axis=0)
    lidi_offsets = np.asarray([0, 4, 13], dtype=np.int64)

    with h5py.File(path, "w") as h5:
        h5.create_dataset("mol_offsets", data=mol_offsets, dtype=np.int64)
        h5.create_dataset("atom_numbers", data=atom_numbers, dtype=np.int64)
        h5.create_dataset("coords", data=coords, dtype=np.float32)
        h5.create_dataset("charges", data=charges, dtype=np.float32)
        h5.create_dataset("lidi_offsets", data=lidi_offsets, dtype=np.int64)
        h5.create_dataset("lidi_values", data=lidi_flat, dtype=np.float32)


def test_lidi_dataset_loading_and_collate(tmp_path):
    h5_path = tmp_path / "tiny_lidi.h5"
    _write_tiny_lidi_h5(h5_path)

    ds = H5SingleMolPretrainDataset(
        h5_files=[str(h5_path)],
        mask_ratio=0.0,
        noise_std=0.0,
        deterministic=True,
        require_lidi=True,
    )
    assert len(ds) == 2

    s0 = ds[0]
    s1 = ds[1]
    assert "lidi_matrix" in s0 and "lidi_matrix" in s1
    assert tuple(s0["lidi_matrix"].shape) == (2, 2)
    assert tuple(s1["lidi_matrix"].shape) == (3, 3)

    batch = collate_fn_pretrain([s0, s1])
    assert "lidi_matrix" in batch
    assert "lidi_valid" in batch
    assert tuple(batch["lidi_matrix"].shape) == (2, 3, 3)
    assert tuple(batch["lidi_valid"].shape) == (2, 3, 3)

    # Padded entries should be invalid.
    assert batch["lidi_valid"][0, 2, :].sum() == 0
    assert batch["lidi_valid"][0, :, 2].sum() == 0
    assert bool(torch.isfinite(batch["lidi_matrix"]).all())
