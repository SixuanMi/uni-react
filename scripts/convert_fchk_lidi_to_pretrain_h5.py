#!/usr/bin/env python3
"""Convert paired Gaussian .fchk + LIDI .txt files into pretrain-ready HDF5.

Output schema (extxyz-compatible + LIDI):
  /mol_offsets      int64  [num_molecules + 1]
  /atom_numbers     int64  [total_atoms]
  /coords           float32 [total_atoms, 3]
  /charges          float32 [total_atoms]
  /lidi_offsets     int64  [num_molecules + 1]
  /lidi_values      float32/float16 [sum_i (N_i * N_i)]
  /electron_count   float32 [num_molecules]  (diag + 0.5 * offdiag)
  /molecular_charge float32 [num_molecules]
  /multiplicity     int32 [num_molecules]
  /sample_id        utf-8 string [num_molecules] (optional)

Examples:
  python scripts/convert_fchk_lidi_to_pretrain_h5.py \
    --fchk-dir /data/RGD1_fchk \
    --lidi-dir /data/RGD1_LIDI \
    --out-prefix data/rgd1_lidi_stage1 \
    --compression gzip --gzip-level 4 --lidi-dtype float16

  python scripts/convert_fchk_lidi_to_pretrain_h5.py \
    --fchk-dir /data/RGD1_fchk \
    --lidi-dir /data/RGD1_LIDI \
    --out-prefix data/rgd1_lidi_stage1 \
    --val-ratio 0.02 \
    --archive-tar-gz
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from tqdm import tqdm


BOHR_TO_ANGSTROM = 0.529177210903
ARRAY_META_RE = re.compile(r"^([IRC])\s+N=\s*(\d+)\s*$")


@dataclass
class ParsedFchk:
    atomic_numbers: np.ndarray  # (N,), int64
    coords_angstrom: np.ndarray  # (N, 3), float32
    charges: np.ndarray  # (N,), float32
    molecular_charge: float
    multiplicity: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert paired .fchk + LIDI .txt files into pretrain HDF5."
    )
    p.add_argument("--fchk-dir", type=str, required=True, help="Directory containing .fchk files.")
    p.add_argument("--lidi-dir", type=str, required=True, help="Directory containing LIDI .txt files.")
    p.add_argument("--fchk-pattern", type=str, default="*.fchk", help="Glob pattern for fchk files.")
    p.add_argument("--lidi-pattern", type=str, default="*.txt", help="Glob pattern for LIDI files.")
    p.add_argument("--no-recursive", action="store_true", help="Disable recursive file search.")

    p.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Output prefix. Produces <prefix>.h5 or <prefix>_train.h5 + <prefix>_val.h5.",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="If > 0, split into train/val by deterministic hash on sample id.",
    )
    p.add_argument("--split-seed", type=int, default=42, help="Seed for deterministic train/val split.")

    p.add_argument(
        "--coords-unit",
        type=str,
        choices=("bohr", "angstrom"),
        default="bohr",
        help="Unit of 'Current cartesian coordinates' in fchk.",
    )
    p.add_argument(
        "--charge-source",
        type=str,
        choices=("mulliken", "zeros"),
        default="mulliken",
        help="Charge source for /charges dataset.",
    )
    p.add_argument(
        "--on-missing-charge",
        type=str,
        choices=("error", "zeros"),
        default="zeros",
        help="Behavior when Mulliken charges are missing and --charge-source mulliken.",
    )
    p.add_argument(
        "--lidi-dtype",
        type=str,
        choices=("float32", "float16"),
        default="float32",
        help="Storage dtype for /lidi_values.",
    )
    p.add_argument(
        "--symmetry-tol",
        type=float,
        default=1e-4,
        help="Allowed max |LIDI - LIDI^T| before optional symmetrization.",
    )
    p.add_argument(
        "--no-symmetrize",
        action="store_true",
        help="Do not symmetrize LIDI matrix when asymmetry exceeds tolerance.",
    )
    p.add_argument(
        "--check-electron-conservation",
        action="store_true",
        help="Check |effective_lidi_electrons-derived_electrons| <= electron_tol for each molecule.",
    )
    p.add_argument(
        "--electron-tol",
        type=float,
        default=3.0,
        help="Tolerance for electron conservation check (absolute).",
    )

    p.add_argument(
        "--compression",
        type=str,
        choices=("none", "lzf", "gzip"),
        default="lzf",
        help="HDF5 compression.",
    )
    p.add_argument("--gzip-level", type=int, default=4, help="gzip level when --compression gzip.")
    p.add_argument("--flush-mols", type=int, default=1024, help="Flush buffer every N molecules.")
    p.add_argument("--max-errors", type=int, default=0, help="Maximum number of skipped bad samples.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    p.add_argument(
        "--archive-tar-gz",
        action="store_true",
        help="Pack generated HDF5 file(s) into <out_prefix>.tar.gz for transfer.",
    )
    p.add_argument(
        "--archive-level",
        type=int,
        default=6,
        help="tar.gz compression level (1-9) when --archive-tar-gz.",
    )
    p.add_argument("--no-sample-id", action="store_true", help="Do not write /sample_id dataset.")
    return p.parse_args()


def _iter_files(root: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(p for p in root.rglob(pattern) if p.is_file())
    return sorted(p for p in root.glob(pattern) if p.is_file())


def _build_stem_map(paths: Sequence[Path], tag: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    dup: Dict[str, List[Path]] = {}
    for p in paths:
        stem = p.stem
        if stem in out:
            dup.setdefault(stem, [out[stem]]).append(p)
        else:
            out[stem] = p
    if dup:
        msg = []
        for stem, ps in sorted(dup.items())[:5]:
            msg.append(f"{stem}: {[str(x) for x in ps]}")
        preview = "\n".join(msg)
        raise ValueError(
            f"Found duplicate {tag} stems. Please make stems unique.\n{preview}"
        )
    return out


def _read_fchk_targeted(path: Path) -> ParsedFchk:
    num_atoms: Optional[int] = None
    molecular_charge: Optional[float] = None
    multiplicity: Optional[int] = None
    atomic_numbers: Optional[np.ndarray] = None
    coords_flat: Optional[np.ndarray] = None
    mulliken: Optional[np.ndarray] = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = iter(f)
        for line in lines:
            if len(line) < 44:
                continue
            name = line[:43].strip()
            meta = line[43:].strip()

            if name == "Number of atoms" and num_atoms is None:
                parts = meta.split()
                if len(parts) >= 2 and parts[0] == "I":
                    num_atoms = int(parts[-1])
                continue

            if name == "Charge" and molecular_charge is None:
                parts = meta.split()
                if len(parts) >= 2 and parts[0] in {"I", "R"}:
                    molecular_charge = float(parts[-1])
                continue

            if name == "Multiplicity" and multiplicity is None:
                parts = meta.split()
                if len(parts) >= 2 and parts[0] == "I":
                    multiplicity = int(parts[-1])
                continue

            if name not in {"Atomic numbers", "Current cartesian coordinates", "Mulliken Charges"}:
                continue

            m = ARRAY_META_RE.match(meta)
            if m is None:
                continue
            kind, n_str = m.group(1), m.group(2)
            count = int(n_str)

            vals: List[str] = []
            while len(vals) < count:
                try:
                    data_line = next(lines)
                except StopIteration as exc:
                    raise ValueError(f"Unexpected EOF while reading '{name}' in {path}") from exc
                vals.extend(data_line.split())

            if kind == "I":
                arr = np.asarray(vals[:count], dtype=np.int64)
            elif kind == "R":
                arr = np.asarray(vals[:count], dtype=np.float64)
            else:
                continue

            if name == "Atomic numbers" and atomic_numbers is None:
                atomic_numbers = arr.astype(np.int64, copy=False)
            elif name == "Current cartesian coordinates" and coords_flat is None:
                coords_flat = arr.astype(np.float64, copy=False)
            elif name == "Mulliken Charges" and mulliken is None:
                mulliken = arr.astype(np.float64, copy=False)

    if num_atoms is None:
        raise ValueError(f"Missing 'Number of atoms' in {path}")
    if atomic_numbers is None:
        raise ValueError(f"Missing 'Atomic numbers' in {path}")
    if coords_flat is None:
        raise ValueError(f"Missing 'Current cartesian coordinates' in {path}")
    if molecular_charge is None:
        raise ValueError(f"Missing 'Charge' in {path}")
    if multiplicity is None:
        raise ValueError(f"Missing 'Multiplicity' in {path}")

    if atomic_numbers.shape[0] != num_atoms:
        raise ValueError(
            f"Atomic numbers length mismatch in {path}: {atomic_numbers.shape[0]} != {num_atoms}"
        )
    if coords_flat.shape[0] != (3 * num_atoms):
        raise ValueError(
            f"Coordinate length mismatch in {path}: {coords_flat.shape[0]} != {3 * num_atoms}"
        )
    coords = coords_flat.reshape(num_atoms, 3).astype(np.float32)

    if mulliken is not None:
        if mulliken.shape[0] != num_atoms:
            raise ValueError(
                f"Mulliken charge length mismatch in {path}: {mulliken.shape[0]} != {num_atoms}"
            )
        charges = mulliken.astype(np.float32)
    else:
        charges = np.empty((0,), dtype=np.float32)

    return ParsedFchk(
        atomic_numbers=atomic_numbers,
        coords_angstrom=coords,
        charges=charges,
        molecular_charge=float(molecular_charge),
        multiplicity=int(multiplicity),
    )


def parse_lidi_txt_matrix(path: Path, n_atoms: int) -> np.ndarray:
    mat = np.full((n_atoms, n_atoms), np.nan, dtype=np.float64)
    current_cols: Optional[List[int]] = None
    row_lines = 0
    in_localization = False
    localization_vals: Dict[int, float] = {}
    loc_pat = re.compile(r"(\d+)\([^)]*\):\s*([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)")

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "Localization index" in line:
                in_localization = True
                continue

            if in_localization:
                for m in loc_pat.finditer(line):
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < n_atoms:
                        localization_vals[idx] = float(m.group(2))
                continue

            if line.startswith("*"):
                continue

            toks = line.split()
            if not toks:
                continue

            if all(tok.isdigit() for tok in toks):
                cols = [int(tok) - 1 for tok in toks]
                if any(c < 0 or c >= n_atoms for c in cols):
                    continue
                current_cols = cols
                continue

            if current_cols is None:
                continue

            if not toks[0].isdigit() or len(toks) < 2:
                continue

            row = int(toks[0]) - 1
            if row < 0 or row >= n_atoms:
                continue

            values: List[float] = []
            ok = True
            for tok in toks[1:]:
                try:
                    values.append(float(tok))
                except ValueError:
                    ok = False
                    break
            if not ok:
                continue

            n_fill = min(len(values), len(current_cols))
            for j in range(n_fill):
                mat[row, current_cols[j]] = values[j]
            row_lines += 1

    if np.isnan(mat).any():
        nan_count = int(np.isnan(mat).sum())
        raise ValueError(
            f"Incomplete LIDI matrix in {path}: {nan_count} entries missing for N={n_atoms}"
        )
    if row_lines < n_atoms:
        raise ValueError(
            f"Too few parsed LIDI row lines in {path}: parsed={row_lines}, expected>={n_atoms}"
        )

    if localization_vals:
        if len(localization_vals) != n_atoms:
            missing = sorted(set(range(n_atoms)) - set(localization_vals))
            raise ValueError(
                f"Incomplete Localization index in {path}: got {len(localization_vals)}/{n_atoms}, "
                f"missing_atom_indices(1-based)={[m + 1 for m in missing[:10]]}"
            )
        for i in range(n_atoms):
            mat[i, i] = localization_vals[i]

    return mat


def _compression_kwargs(
    compression: str,
    gzip_level: int,
) -> Dict[str, object]:
    if compression == "none":
        return {}
    if compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    return {"compression": "gzip", "compression_opts": int(gzip_level), "shuffle": True}


class H5PackedWriter:
    def __init__(
        self,
        out_path: Path,
        *,
        compression: str,
        gzip_level: int,
        lidi_dtype: str,
        flush_mols: int,
        write_sample_id: bool,
    ) -> None:
        self.out_path = out_path
        self.flush_mols = int(flush_mols)
        self._write_sample_id = bool(write_sample_id)
        self._lidi_dtype = np.float16 if lidi_dtype == "float16" else np.float32

        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(out_path, "w")

        ck = _compression_kwargs(compression=compression, gzip_level=gzip_level)
        self.h5.create_dataset("mol_offsets", data=np.asarray([0], dtype=np.int64), maxshape=(None,), **ck)
        self.h5.create_dataset("atom_numbers", shape=(0,), maxshape=(None,), dtype=np.int64, **ck)
        self.h5.create_dataset("coords", shape=(0, 3), maxshape=(None, 3), dtype=np.float32, **ck)
        self.h5.create_dataset("charges", shape=(0,), maxshape=(None,), dtype=np.float32, **ck)
        self.h5.create_dataset("lidi_offsets", data=np.asarray([0], dtype=np.int64), maxshape=(None,), **ck)
        self.h5.create_dataset("lidi_values", shape=(0,), maxshape=(None,), dtype=self._lidi_dtype, **ck)
        self.h5.create_dataset("electron_count", shape=(0,), maxshape=(None,), dtype=np.float32, **ck)
        self.h5.create_dataset("molecular_charge", shape=(0,), maxshape=(None,), dtype=np.float32, **ck)
        self.h5.create_dataset("multiplicity", shape=(0,), maxshape=(None,), dtype=np.int32, **ck)
        if self._write_sample_id:
            str_dtype = h5py.string_dtype(encoding="utf-8")
            self.h5.create_dataset("sample_id", shape=(0,), maxshape=(None,), dtype=str_dtype)

        self.h5.attrs["schema"] = "extxyz_with_lidi_v1"
        self.h5.attrs["coords_unit"] = "angstrom"
        self.h5.attrs["lidi_dtype"] = str(self._lidi_dtype.__name__)

        self._buf_ids: List[str] = []
        self._buf_n_atoms: List[int] = []
        self._buf_z: List[np.ndarray] = []
        self._buf_r: List[np.ndarray] = []
        self._buf_q: List[np.ndarray] = []
        self._buf_lidi_flat: List[np.ndarray] = []
        self._buf_elec: List[float] = []
        self._buf_mol_charge: List[float] = []
        self._buf_mult: List[int] = []
        self.n_written = 0

    def append(
        self,
        *,
        sample_id: str,
        atomic_numbers: np.ndarray,
        coords: np.ndarray,
        charges: np.ndarray,
        lidi_matrix: np.ndarray,
        electron_count: float,
        molecular_charge: float,
        multiplicity: int,
    ) -> None:
        n = int(atomic_numbers.shape[0])
        if coords.shape != (n, 3):
            raise ValueError(f"coords shape mismatch for {sample_id}: {coords.shape} vs ({n}, 3)")
        if charges.shape != (n,):
            raise ValueError(f"charges shape mismatch for {sample_id}: {charges.shape} vs ({n},)")
        if lidi_matrix.shape != (n, n):
            raise ValueError(f"lidi shape mismatch for {sample_id}: {lidi_matrix.shape} vs ({n}, {n})")

        self._buf_ids.append(sample_id)
        self._buf_n_atoms.append(n)
        self._buf_z.append(atomic_numbers.astype(np.int64, copy=False))
        self._buf_r.append(coords.astype(np.float32, copy=False))
        self._buf_q.append(charges.astype(np.float32, copy=False))
        self._buf_lidi_flat.append(lidi_matrix.reshape(-1).astype(self._lidi_dtype, copy=False))
        self._buf_elec.append(float(electron_count))
        self._buf_mol_charge.append(float(molecular_charge))
        self._buf_mult.append(int(multiplicity))

        if len(self._buf_n_atoms) >= self.flush_mols:
            self.flush()

    def flush(self) -> None:
        if not self._buf_n_atoms:
            return

        n_atoms_arr = np.asarray(self._buf_n_atoms, dtype=np.int64)
        z_cat = np.concatenate(self._buf_z, axis=0)
        r_cat = np.concatenate(self._buf_r, axis=0)
        q_cat = np.concatenate(self._buf_q, axis=0)
        lidi_cat = np.concatenate(self._buf_lidi_flat, axis=0)
        elec_arr = np.asarray(self._buf_elec, dtype=np.float32)
        n_mols = int(n_atoms_arr.shape[0])

        atom_ds = self.h5["atom_numbers"]
        r_ds = self.h5["coords"]
        q_ds = self.h5["charges"]
        lidi_ds = self.h5["lidi_values"]
        e_ds = self.h5["electron_count"]
        c_ds = self.h5["molecular_charge"]
        m_ds = self.h5["multiplicity"]
        mol_offsets_ds = self.h5["mol_offsets"]
        lidi_offsets_ds = self.h5["lidi_offsets"]

        atom_start = atom_ds.shape[0]
        atom_end = atom_start + z_cat.shape[0]
        atom_ds.resize((atom_end,))
        atom_ds[atom_start:atom_end] = z_cat
        r_ds.resize((atom_end, 3))
        r_ds[atom_start:atom_end, :] = r_cat
        q_ds.resize((atom_end,))
        q_ds[atom_start:atom_end] = q_cat

        lidi_start = lidi_ds.shape[0]
        lidi_end = lidi_start + lidi_cat.shape[0]
        lidi_ds.resize((lidi_end,))
        lidi_ds[lidi_start:lidi_end] = lidi_cat

        e_start = e_ds.shape[0]
        e_end = e_start + n_mols
        e_ds.resize((e_end,))
        e_ds[e_start:e_end] = elec_arr

        c_arr = np.asarray(self._buf_mol_charge, dtype=np.float32)
        c_start = c_ds.shape[0]
        c_end = c_start + n_mols
        c_ds.resize((c_end,))
        c_ds[c_start:c_end] = c_arr

        m_arr = np.asarray(self._buf_mult, dtype=np.int32)
        m_start = m_ds.shape[0]
        m_end = m_start + n_mols
        m_ds.resize((m_end,))
        m_ds[m_start:m_end] = m_arr

        mol_last = int(mol_offsets_ds[-1])
        mol_offsets_new = mol_last + np.cumsum(n_atoms_arr, dtype=np.int64)
        mol_offsets_ds.resize((mol_offsets_ds.shape[0] + n_mols,))
        mol_offsets_ds[-n_mols:] = mol_offsets_new

        lidi_last = int(lidi_offsets_ds[-1])
        lidi_offsets_new = lidi_last + np.cumsum(n_atoms_arr * n_atoms_arr, dtype=np.int64)
        lidi_offsets_ds.resize((lidi_offsets_ds.shape[0] + n_mols,))
        lidi_offsets_ds[-n_mols:] = lidi_offsets_new

        if self._write_sample_id:
            sid_ds = self.h5["sample_id"]
            sid_start = sid_ds.shape[0]
            sid_end = sid_start + n_mols
            sid_ds.resize((sid_end,))
            sid_ds[sid_start:sid_end] = np.asarray(self._buf_ids, dtype=object)

        self.n_written += n_mols
        self._buf_ids.clear()
        self._buf_n_atoms.clear()
        self._buf_z.clear()
        self._buf_r.clear()
        self._buf_q.clear()
        self._buf_lidi_flat.clear()
        self._buf_elec.clear()
        self._buf_mol_charge.clear()
        self._buf_mult.clear()

    def close(self) -> None:
        self.flush()
        self.h5.attrs["num_molecules"] = int(self.n_written)
        self.h5.close()


def _choose_split(sample_id: str, val_ratio: float, seed: int) -> str:
    if val_ratio <= 0:
        return "train"
    payload = f"{sample_id}|{seed}".encode("utf-8")
    h = hashlib.blake2b(payload, digest_size=8).digest()
    u = int.from_bytes(h, byteorder="little", signed=False) / float(2**64)
    return "val" if u < val_ratio else "train"


def _archive_tar_gz(archive_path: Path, files: Sequence[Path], level: int) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w:gz", compresslevel=int(level)) as tar:
        for p in files:
            tar.add(p, arcname=p.name)


def _effective_electron_count_from_matrix(mat: np.ndarray) -> float:
    diag = float(np.trace(mat))
    total = float(np.sum(mat))
    return 0.5 * (total + diag)


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError(f"--val-ratio must be in [0,1), got {args.val_ratio}")
    if args.flush_mols <= 0:
        raise ValueError(f"--flush-mols must be > 0, got {args.flush_mols}")
    if not (1 <= args.archive_level <= 9):
        raise ValueError(f"--archive-level must be in [1,9], got {args.archive_level}")

    recursive = not args.no_recursive
    fchk_dir = Path(args.fchk_dir).expanduser().resolve()
    lidi_dir = Path(args.lidi_dir).expanduser().resolve()
    if not fchk_dir.exists():
        raise FileNotFoundError(f"fchk dir not found: {fchk_dir}")
    if not lidi_dir.exists():
        raise FileNotFoundError(f"lidi dir not found: {lidi_dir}")

    fchk_files = _iter_files(fchk_dir, args.fchk_pattern, recursive=recursive)
    lidi_files = _iter_files(lidi_dir, args.lidi_pattern, recursive=recursive)
    if not fchk_files:
        raise ValueError(f"No fchk files found in {fchk_dir} with pattern {args.fchk_pattern!r}")
    if not lidi_files:
        raise ValueError(f"No LIDI txt files found in {lidi_dir} with pattern {args.lidi_pattern!r}")

    fchk_map = _build_stem_map(fchk_files, tag="fchk")
    lidi_map = _build_stem_map(lidi_files, tag="lidi")
    stems = sorted(set(fchk_map) & set(lidi_map))
    if not stems:
        raise ValueError("No paired samples found by filename stem intersection.")

    print(f"[scan] fchk={len(fchk_map)}  lidi={len(lidi_map)}  paired={len(stems)}")

    out_prefix = Path(args.out_prefix).expanduser().resolve()
    out_files: List[Path] = []

    if args.val_ratio > 0:
        out_train = out_prefix.with_name(out_prefix.name + "_train.h5")
        out_val = out_prefix.with_name(out_prefix.name + "_val.h5")
        out_files.extend([out_train, out_val])
        for p in out_files:
            if p.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists: {p}. Use --overwrite to replace.")
        train_writer = H5PackedWriter(
            out_train,
            compression=args.compression,
            gzip_level=args.gzip_level,
            lidi_dtype=args.lidi_dtype,
            flush_mols=args.flush_mols,
            write_sample_id=not args.no_sample_id,
        )
        val_writer = H5PackedWriter(
            out_val,
            compression=args.compression,
            gzip_level=args.gzip_level,
            lidi_dtype=args.lidi_dtype,
            flush_mols=args.flush_mols,
            write_sample_id=not args.no_sample_id,
        )
    else:
        out_all = out_prefix.with_suffix(".h5")
        out_files.append(out_all)
        if out_all.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {out_all}. Use --overwrite to replace.")
        train_writer = H5PackedWriter(
            out_all,
            compression=args.compression,
            gzip_level=args.gzip_level,
            lidi_dtype=args.lidi_dtype,
            flush_mols=args.flush_mols,
            write_sample_id=not args.no_sample_id,
        )
        val_writer = None

    n_ok = 0
    n_err = 0
    n_asym_fixed = 0
    max_asym = 0.0
    electron_diff_max = 0.0

    try:
        for stem in tqdm(stems, desc="convert", unit="mol"):
            fchk_path = fchk_map[stem]
            lidi_path = lidi_map[stem]
            sample_id = stem

            try:
                parsed = _read_fchk_targeted(fchk_path)
                z = parsed.atomic_numbers
                r = parsed.coords_angstrom
                q = parsed.charges
                n_atoms = int(z.shape[0])

                if args.coords_unit == "bohr":
                    r = (r.astype(np.float64) * BOHR_TO_ANGSTROM).astype(np.float32)
                else:
                    r = r.astype(np.float32, copy=False)

                if args.charge_source == "zeros":
                    q = np.zeros((n_atoms,), dtype=np.float32)
                else:
                    if q.shape[0] == 0:
                        if args.on_missing_charge == "error":
                            raise ValueError(f"Missing Mulliken Charges in {fchk_path}")
                        q = np.zeros((n_atoms,), dtype=np.float32)
                    else:
                        q = q.astype(np.float32, copy=False)

                lidi = parse_lidi_txt_matrix(lidi_path, n_atoms=n_atoms)
                asym = float(np.max(np.abs(lidi - lidi.T)))
                max_asym = max(max_asym, asym)
                if asym > args.symmetry_tol:
                    if args.no_symmetrize:
                        raise ValueError(
                            f"LIDI asymmetry too large in {lidi_path}: max|A-A^T|={asym:.3e}"
                        )
                    lidi = 0.5 * (lidi + lidi.T)
                    n_asym_fixed += 1

                if not np.isfinite(lidi).all():
                    raise ValueError(f"Non-finite LIDI values in {lidi_path}")

                # Derive electron count from total nuclear charge and molecular charge.
                electron_count = float(np.sum(z, dtype=np.int64) - parsed.molecular_charge)
                matrix_electron_count = _effective_electron_count_from_matrix(lidi)
                electron_diff = abs(matrix_electron_count - electron_count)
                electron_diff_max = max(electron_diff_max, electron_diff)
                if args.check_electron_conservation and electron_diff > args.electron_tol:
                    raise ValueError(
                        f"Electron conservation check failed for {sample_id}: "
                        f"effective_electrons={matrix_electron_count:.6f}, "
                        f"derived_electrons={electron_count:.6f}, diff={electron_diff:.6f}"
                    )

                target = _choose_split(sample_id, val_ratio=args.val_ratio, seed=args.split_seed)
                writer = train_writer if (target == "train" or val_writer is None) else val_writer
                writer.append(
                    sample_id=sample_id,
                    atomic_numbers=z,
                    coords=r,
                    charges=q,
                    lidi_matrix=lidi,
                    electron_count=electron_count,
                    molecular_charge=parsed.molecular_charge,
                    multiplicity=parsed.multiplicity,
                )
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                n_err += 1
                if args.max_errors <= 0 or n_err > args.max_errors:
                    raise RuntimeError(
                        f"Failed at sample {sample_id} ({fchk_path.name}, {lidi_path.name}). "
                        f"errors={n_err}"
                    ) from exc
                print(f"[warn] skip {sample_id}: {exc}")
    finally:
        train_writer.close()
        if val_writer is not None:
            val_writer.close()

    if args.archive_tar_gz:
        archive_path = out_prefix.with_suffix(".tar.gz")
        _archive_tar_gz(archive_path, files=out_files, level=args.archive_level)
        print(f"[archive] {archive_path}")

    print("[done]")
    print(f"  converted={n_ok}")
    print(f"  skipped={n_err}")
    print(f"  max_lidi_asym_before_fix={max_asym:.6e}")
    print(f"  asym_fixed_count={n_asym_fixed}")
    print(f"  max_abs(effective_lidi_electrons-derived_electrons)={electron_diff_max:.6e}")
    for p in out_files:
        size_gb = p.stat().st_size / (1024**3) if p.exists() else math.nan
        print(f"  output={p}  size_gb={size_gb:.3f}")


if __name__ == "__main__":
    main()
