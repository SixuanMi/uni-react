# Configuration Files

This directory contains training configurations for different encoder architectures.

## Directory Structure

```
configs/
├── single_mol/          # SingleMol encoder (SE(3) equivariant MPNN)
│   ├── geometric.yaml   # Stage 1: Geometric structure pretraining
│   ├── cdft.yaml        # Stage 2a: Electronic structure (VIP/VEA + Fukui)
│   ├── density.yaml     # Stage 2b: Electron density prediction
│   └── reaction.yaml    # Stage 3: Reaction triplet pretraining
│
├── reacformer_se3/      # ReacFormer SE(3) (Full SE(3) with CG tensors)
│   ├── geometric.yaml   # Stage 1: Geometric structure pretraining
│   ├── cdft.yaml        # Stage 2a: Electronic structure (VIP/VEA + Fukui)
│   ├── density.yaml     # Stage 2b: Electron density prediction
│   └── reaction.yaml    # Stage 3: Reaction triplet pretraining
│
├── reacformer_so2/      # ReacFormer SO(2) (Lighter, faster)
│   ├── geometric.yaml   # Stage 1: Geometric structure pretraining
│   ├── cdft.yaml        # Stage 2a: Electronic structure (VIP/VEA + Fukui)
│   ├── density.yaml     # Stage 2b: Electron density prediction
│   └── reaction.yaml    # Stage 3: Reaction triplet pretraining
│
├── finetune_qm9_gap.yaml     # QM9 fine-tuning (HOMO-LUMO gap)
└── finetune_qm9_all.yaml     # QM9 fine-tuning (all 12 targets)
```

## Three-Stage Pretraining Pipeline

### Stage 1: Geometric Structure
- **Task**: Atom masking + coordinate denoising + charge prediction
- **Data**: GDB13 molecular datasets
- **Config**: `*/geometric.yaml`

### Stage 1 LIDI Ablations (SingleMol templates)
- `single_mol/geometric_ablation_baseline.yaml`:
  atom-mask + coord-denoise (no charge)
- `single_mol/geometric_ablation_lidi_node.yaml`:
  baseline + LIDI diagonal (node) supervision
- `single_mol/geometric_ablation_lidi_edge.yaml`:
  baseline + LIDI off-diagonal (edge) supervision
- `single_mol/geometric_ablation_lidi_full.yaml`:
  baseline + node + edge + conservation supervision

Recommended data prep flow for large RGD1-style files:
1. Extract minimal fields from huge `fchk` + pair with LIDI txt:
   - `scripts/extract_minimal_fchk_lidi.sh`
2. Convert minimal files to stage-1 HDF5:
   - `scripts/convert_fchk_lidi_to_pretrain_h5.py`

LIDI reference docs:
- `docs/lidi_h5_format.md` (production format and converter behavior)
- `scripts/make_lidi_example_h5.py` (tiny synthetic example generator)

### Stage 2a: Electronic Structure
- **Task**: VIP/VEA + Fukui indices prediction
- **Data**: CDFT calculations (RGD1_CDFT for train, t1x_CDFT for val)
- **Config**: `*/cdft.yaml`
- **Init**: Load checkpoint from Stage 1

### Stage 2b: Electron Density (Alternative to 2a)
- **Task**: Electron density prediction
- **Data**: EDBench density datasets
- **Config**: `*/density.yaml`
- **Init**: Load checkpoint from Stage 1

### Stage 3: Reaction Triplet
- **Task**: Reaction consistency learning (reactant + reagent → product)
- **Data**: Reaction triplet datasets (RGD1_react.h5 for train, t1x_react.h5 for val)
- **Config**: `*/reaction.yaml`
- **Init**: Load checkpoint from Stage 2a or 2b

## Model Comparison

| Model | Type | Speed | Memory | Accuracy |
|-------|------|-------|--------|----------|
| **single_mol** | SE(3) MPNN | Fast | Low | Good |
| **reacformer_se3** | Full SE(3) CG | Slow | High | Best |
| **reacformer_so2** | SO(2) | Medium | Medium | Good |

## Usage Examples

### SingleMol - Full Pipeline
```bash
# Stage 1: Geometric
python -m uni_react.train_pretrain_geometric --config configs/single_mol/geometric.yaml

# Stage 2a: Electronic (VIP/VEA + Fukui)
python -m uni_react.train_pretrain_cdft --config configs/single_mol/cdft.yaml \
  --init_ckpt runs/single_mol_geometric/best.pt

# Stage 2b: Density (Alternative to 2a)
python -m uni_react.train_pretrain_density --config configs/single_mol/density.yaml \
  --init_ckpt runs/single_mol_geometric/best.pt

# Stage 3: Reaction (use checkpoint from 2a or 2b)
python -m uni_react.train_pretrain_reaction --config configs/single_mol/reaction.yaml \
  --init_ckpt runs/single_mol_cdft/best.pt
```

### ReacFormer SE(3) - Full Pipeline
```bash
# Stage 1: Geometric
python -m uni_react.train_pretrain_geometric --config configs/reacformer_se3/geometric.yaml

# Stage 2a: Electronic
python -m uni_react.train_pretrain_cdft --config configs/reacformer_se3/cdft.yaml \
  --init_ckpt runs/reacformer_se3_geometric/best.pt

# Stage 2b: Density (Alternative)
python -m uni_react.train_pretrain_density --config configs/reacformer_se3/density.yaml \
  --init_ckpt runs/reacformer_se3_geometric/best.pt

# Stage 3: Reaction
python -m uni_react.train_pretrain_reaction --config configs/reacformer_se3/reaction.yaml \
  --init_ckpt runs/reacformer_se3_cdft/best.pt
```

### ReacFormer SO(2) - Full Pipeline
```bash
# Stage 1: Geometric
python -m uni_react.train_pretrain_geometric --config configs/reacformer_so2/geometric.yaml

# Stage 2a: Electronic
python -m uni_react.train_pretrain_cdft --config configs/reacformer_so2/cdft.yaml \
  --init_ckpt runs/reacformer_so2_geometric/best.pt

# Stage 2b: Density (Alternative)
python -m uni_react.train_pretrain_density --config configs/reacformer_so2/density.yaml \
  --init_ckpt runs/reacformer_so2_geometric/best.pt

# Stage 3: Reaction
python -m uni_react.train_pretrain_reaction --config configs/reacformer_so2/reaction.yaml \
  --init_ckpt runs/reacformer_so2_cdft/best.pt
```

## Multi-GPU Training

Use `torchrun` for distributed training:

```bash
torchrun --nproc_per_node=8 -m uni_react.train_pretrain_geometric \
  --config configs/reacformer_se3/geometric.yaml
```

## Override Config Parameters

Any config parameter can be overridden via CLI:

```bash
python -m uni_react.train_pretrain_geometric \
  --config configs/single_mol/geometric.yaml \
  --lr 5e-5 \
  --batch_size 64 \
  --epochs 30
```

## Key Configuration Parameters

### Model Architecture
- `encoder_type`: Model type (`single_mol`, `reacformer_se3`, `reacformer_so2`)
- `emb_dim`: Embedding dimension (default: 256)
- `se3_layer`: Number of equivariant layers (default: 4)
- `heads`: Number of attention heads (default: 8)

### Training
- `lr`: Stage-1/2 learning rate
- `backbone_lr` / `head_lr`: Stage-3 reaction and QM9 split learning rates
- `batch_size`: Batch size (128 for SingleMol, 64 for SE(3), 96 for SO(2))
- `epochs`: Number of training epochs (default: 20)
- `warmup_steps`: Warmup steps (1000 for SingleMol/SO(2), 2000 for SE(3))

### Data Augmentation
- `mask_ratio`: Masking ratio for geometric training (default: 0.15)
- `noise_std`: Coordinate noise std for geometric training (default: 0.1)
- Set both to 0.0 for electronic structure training

## Notes

- **Stage 1 (Geometric)** is the foundation and should always be trained first
- **Stage 2** has two options:
  - **2a (Electronic)**: VIP/VEA + Fukui indices using CDFT data
  - **2b (Density)**: Electron density prediction using EDBench data
  - Choose one based on your downstream task requirements
- **Stage 3 (Reaction)** can use checkpoint from either Stage 2a or 2b
- Each stage loads the checkpoint from the previous stage for warm-start
- **Data split convention**: RGD1 = train, t1x = val
- ReacFormer SE(3) is the most accurate but slowest and most memory-intensive
- ReacFormer SO(2) offers a good balance between speed and accuracy
- SingleMol is the fastest and most memory-efficient
