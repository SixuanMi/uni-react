# LIDI Stage-1 Input HDF5 Format (Reference)

This document describes a practical HDF5 format for stage-1 pretraining with optional LIDI supervision.

## 1) Required fields (baseline stage-1)

Use one of the two existing schemas already supported by `H5SingleMolPretrainDataset`:

### Schema A: `extxyz` style

- `/mol_offsets` : `int64`, shape `(num_molecules + 1,)`
- `/atom_numbers` : `int64`, shape `(total_atoms,)`
- `/coords` : `float32`, shape `(total_atoms, 3)`
- `/charges` : `float32`, shape `(total_atoms,)` (optional)

For molecule `i`, atom slice is:

- `start = mol_offsets[i]`
- `end = mol_offsets[i + 1]`
- `N = end - start`

### Schema B: `stable_gen` style

- `/frames/offsets` : `int64`, shape `(num_molecules + 1,)`
- `/atoms/Z` : `int64`, shape `(total_atoms,)`
- `/atoms/R` : `float32`, shape `(total_atoms, 3)`
- `/atoms/q` : `float32`, shape `(total_atoms,)` (optional)

## 2) Optional LIDI labels (for ablation B/C/D)

The loader supports either of the following:

### Option 1: Flattened LIDI with offsets (recommended for variable-size molecules)

- `/lidi_offsets` or `/frames/lidi_offsets` : `int64`, shape `(num_molecules + 1,)`
- `/lidi_values` or `/frames/lidi_values` : `float32`, shape `(sum_i N_i * N_i,)`

For molecule `i`:

- `ls = lidi_offsets[i]`
- `le = lidi_offsets[i + 1]`
- `flat = lidi_values[ls:le]`
- `lidi_matrix = flat.reshape(N_i, N_i)`

`N_i` must match the atom count from molecule offsets.

### Option 2: 3-D padded LIDI matrix

- `/lidi_matrix` or `/frames/lidi_matrix` : `float32`, shape `(num_molecules, max_atoms, max_atoms)`

For molecule `i`, the valid part is `lidi_matrix[i, :N_i, :N_i]`.

## 3) Optional electron count (for conservation loss)

If you want explicit conservation targets, use:

- `/electron_count` : `float32`, shape `(num_molecules,)`

For symmetric LIDI matrices, the effective electron count is:
`diag_sum + 0.5 * offdiag_sum`.

If omitted, conservation loss falls back to this effective count computed from target LIDI.

## 4) Notes

- Keep all numeric arrays finite (`NaN`/`Inf` are rejected).
- `atom_numbers` should be within `atom_vocab_size`.
- For ablation baseline (A), LIDI datasets can be absent.
- For LIDI ablations (B/C/D), include LIDI datasets in every train/val shard.

## 5) fchk + LIDI txt conversion script

Use:

- `scripts/convert_fchk_lidi_to_pretrain_h5.py`

Example:

```bash
python scripts/convert_fchk_lidi_to_pretrain_h5.py \
  --fchk-dir /path/to/RGD1_fchk \
  --lidi-dir /path/to/RGD1_LIDI \
  --out-prefix /path/to/rgd1_stage1 \
  --compression gzip --gzip-level 4 --lidi-dtype float16
```

Behavior details:

- Reads from `fchk`: `Atomic numbers`, `Current cartesian coordinates`, `Mulliken Charges` (optional), `Charge`, `Multiplicity`.
- Does **not** read `Number of electrons`; instead derives electron count as `sum(Z) - Charge`.
- For LIDI txt, matrix blocks provide pair terms and the final `Localization index` section provides the **true diagonal**.
- Stored `electron_count` uses effective electron counting for symmetric pair matrices:
  `electron_count = diag_sum + 0.5 * offdiag_sum`.
- Output also includes:
  - `/molecular_charge` : `float32`, shape `(num_molecules,)`
  - `/multiplicity` : `int32`, shape `(num_molecules,)`
