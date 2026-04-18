#!/usr/bin/env python3
"""Create a tiny example HDF5 file for stage-1 + LIDI pretraining.

Example:
  python scripts/make_lidi_example_h5.py --out /tmp/lidi_example.h5 --num_mols 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _build_lidi_matrix(
    atomic_numbers: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct a synthetic dense LIDI matrix for demonstration.

    - Diagonal encodes per-atom electron-like magnitude.
    - Off-diagonal encodes weak pair interactions.
    - Matrix is symmetric and finite.
    """
    n = int(atomic_numbers.shape[0])
    diag = atomic_numbers.astype(np.float32) * 0.15 + 0.5
    rand = rng.uniform(low=0.0, high=0.1, size=(n, n)).astype(np.float32)
    offdiag = 0.5 * (rand + rand.T)
    np.fill_diagonal(offdiag, 0.0)
    lidi = offdiag
    np.fill_diagonal(lidi, diag)
    return lidi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a minimal extxyz-style HDF5 with optional LIDI labels."
    )
    parser.add_argument("--out", type=str, required=True, help="Output .h5 path.")
    parser.add_argument("--num_mols", type=int, default=16, help="Number of molecules.")
    parser.add_argument("--min_atoms", type=int, default=4, help="Min atoms per molecule.")
    parser.add_argument("--max_atoms", type=int, default=12, help="Max atoms per molecule.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.num_mols <= 0:
        raise ValueError("--num_mols must be > 0")
    if args.min_atoms <= 0:
        raise ValueError("--min_atoms must be > 0")
    if args.max_atoms < args.min_atoms:
        raise ValueError("--max_atoms must be >= --min_atoms")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    z_list = []
    r_list = []
    q_list = []
    mol_offsets = [0]

    lidi_flat_list = []
    lidi_offsets = [0]
    electron_count = []

    atom_pool = np.asarray([1, 6, 7, 8, 9, 16], dtype=np.int64)
    for _ in range(args.num_mols):
        n = int(rng.integers(args.min_atoms, args.max_atoms + 1))
        z = rng.choice(atom_pool, size=n, replace=True).astype(np.int64)
        r = rng.normal(loc=0.0, scale=1.0, size=(n, 3)).astype(np.float32)
        q = rng.normal(loc=0.0, scale=0.1, size=(n,)).astype(np.float32)

        lidi = _build_lidi_matrix(z, rng=rng)

        z_list.append(z)
        r_list.append(r)
        q_list.append(q)

        mol_offsets.append(mol_offsets[-1] + n)

        lidi_flat = lidi.reshape(-1)
        lidi_flat_list.append(lidi_flat.astype(np.float32))
        lidi_offsets.append(lidi_offsets[-1] + int(lidi_flat.shape[0]))
        electron_count.append(np.float32(lidi.sum()))

    atom_numbers = np.concatenate(z_list, axis=0)
    coords = np.concatenate(r_list, axis=0)
    charges = np.concatenate(q_list, axis=0)
    mol_offsets_arr = np.asarray(mol_offsets, dtype=np.int64)

    lidi_values = np.concatenate(lidi_flat_list, axis=0)
    lidi_offsets_arr = np.asarray(lidi_offsets, dtype=np.int64)
    electron_count_arr = np.asarray(electron_count, dtype=np.float32)

    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("mol_offsets", data=mol_offsets_arr, dtype=np.int64)
        h5.create_dataset("atom_numbers", data=atom_numbers, dtype=np.int64)
        h5.create_dataset("coords", data=coords, dtype=np.float32)
        h5.create_dataset("charges", data=charges, dtype=np.float32)

        # LIDI (offset + flattened values)
        h5.create_dataset("lidi_offsets", data=lidi_offsets_arr, dtype=np.int64)
        h5.create_dataset("lidi_values", data=lidi_values, dtype=np.float32)
        h5.create_dataset("electron_count", data=electron_count_arr, dtype=np.float32)

        h5.attrs["schema"] = "extxyz_with_lidi_v1"
        h5.attrs["num_molecules"] = int(args.num_mols)

    print(f"[saved] {out_path}")
    print(f"[summary] molecules={args.num_mols}, total_atoms={int(atom_numbers.shape[0])}")


if __name__ == "__main__":
    main()
