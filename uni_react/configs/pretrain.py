"""Dataclass schema for pretraining runs."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PretrainConfig:
    """All hyper-parameters for a single pretraining run.

    Every field has a sensible default so that a minimal YAML only needs to
    specify the fields that differ from the defaults.
    """

    # ------------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------------
    train_mode: str = "geometric_structure"
    """One of ``geometric_structure``, ``cdft`` or legacy ``electronic_structure``."""

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_h5: List[str] = field(default_factory=list)
    """List of HDF5 training files / directories / globs."""

    val_h5: Optional[List[str]] = None
    """Validation HDF5 files (optional)."""

    batch_size: int = 128
    num_workers: int = 16

    # ------------------------------------------------------------------
    # Data augmentation
    # ------------------------------------------------------------------
    mask_ratio: float = 0.15
    mask_token_id: int = 94
    min_masked: int = 1
    max_masked: int = 0
    """0 means no upper bound."""
    noise_std: float = 0.1
    no_center_coords: bool = False
    no_recenter_noisy: bool = False

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    encoder_type: str = "single_mol"
    """Encoder type: single_mol, reacformer_se3, or reacformer_so2."""
    
    emb_dim: int = 256
    inv_layer: int = 2
    se3_layer: int = 4
    heads: int = 8
    atom_vocab_size: int = 128
    cutoff: float = 5.0
    num_kernel: int = 128
    path_dropout: float = 0.1
    activation_dropout: float = 0.1
    attn_dropout: float = 0.1

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    epochs: int = 20
    lr: float = 1e-4
    descriptor_lr: Optional[float] = None
    task_lr: Optional[float] = None
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"
    """One of ``cosine``, ``linear``, or ``none``."""

    # ------------------------------------------------------------------
    # Loss weights (geometric_structure mode)
    # ------------------------------------------------------------------
    geometric_subtasks: List[str] = field(
        default_factory=lambda: ["atom_mask", "coord_denoise", "charge"]
    )
    """Geometric subtasks to enable: atom_mask, coord_denoise, charge, lidi."""

    atom_weight: float = 1.0
    coord_weight: float = 1.0
    charge_weight: float = 1.0
    lidi_node_weight: float = 0.0
    lidi_edge_weight: float = 0.0
    lidi_conservation_weight: float = 0.0
    lidi_reduce: str = "global"
    """Reduction for LIDI node/edge losses: ``global`` or ``per_molecule``."""
    loss_balance_mode: str = "fixed"
    """LIDI loss balancing mode: ``fixed`` or ``ema``."""
    loss_ema_beta: float = 0.98
    """EMA decay for dynamic loss balancing when ``loss_balance_mode=ema``."""
    loss_balance_eps: float = 1e-6
    """Numerical floor for balanced-loss denominator."""
    lidi_edge_transform: str = "none"
    """Optional transform for edge labels/predictions: ``none`` or ``asinh``."""
    lidi_edge_scale: float = 1e-2
    """Scale used by ``asinh(x / scale)`` edge transform."""
    lidi_hidden_dim: int = 256
    """Hidden dimension for the LIDI prediction head."""

    # Optional LIDI data-path overrides in HDF5
    lidi_matrix_key: Optional[str] = None
    lidi_offsets_key: Optional[str] = None
    lidi_values_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Loss weights (electronic_structure mode)
    # ------------------------------------------------------------------
    vip_vea_weight: float = 1.0
    fukui_weight: float = 1.0
    vip_vea_keys: List[str] = field(default_factory=lambda: ["vip", "vea"])
    fukui_keys: List[str] = field(default_factory=lambda: ["f_plus", "f_minus", "f_zero"])

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    device: str = "cuda"
    seed: int = 42
    out_dir: str = ""
    save_every: int = 5
    save_every_steps: int = 0
    """Save checkpoint every N steps (0 = disabled, use save_every instead)."""
    log_interval: int = 100
    """Log batch metrics every N steps (0 = disabled)."""
    log_file: str = "train.log"
    save_optimizer: bool = True
    init_ckpt: Optional[str] = None
    init_strict: bool = False
    restart: Optional[str] = None
    restart_ignore_config: bool = False

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Architecture validation
        if self.emb_dim <= 0:
            raise ValueError(f"emb_dim must be > 0, got {self.emb_dim}")
        
        if self.inv_layer < 1:
            raise ValueError(f"inv_layer must be >= 1, got {self.inv_layer}")
        
        if self.se3_layer < 0:
            raise ValueError(f"se3_layer must be >= 0, got {self.se3_layer}")
        
        if self.heads <= 0:
            raise ValueError(f"heads must be > 0, got {self.heads}")
        
        if self.emb_dim % self.heads != 0:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) must be divisible by heads ({self.heads})"
            )
        
        if self.atom_vocab_size <= 0:
            raise ValueError(f"atom_vocab_size must be > 0, got {self.atom_vocab_size}")
        
        if self.cutoff <= 0:
            raise ValueError(f"cutoff must be > 0, got {self.cutoff}")
        
        if self.num_kernel <= 0:
            raise ValueError(f"num_kernel must be > 0, got {self.num_kernel}")
        
        # Data augmentation validation
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1], got {self.mask_ratio}")
        
        if self.mask_token_id < 0:
            raise ValueError(f"mask_token_id must be >= 0, got {self.mask_token_id}")
        
        if self.mask_token_id >= self.atom_vocab_size:
            raise ValueError(
                f"mask_token_id ({self.mask_token_id}) must be < atom_vocab_size ({self.atom_vocab_size})"
            )
        
        if self.min_masked < 0:
            raise ValueError(f"min_masked must be >= 0, got {self.min_masked}")
        
        if self.max_masked < 0:
            raise ValueError(f"max_masked must be >= 0, got {self.max_masked}")
        
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")
        
        # Dropout validation
        if not 0.0 <= self.path_dropout <= 1.0:
            raise ValueError(f"path_dropout must be in [0, 1], got {self.path_dropout}")
        
        if not 0.0 <= self.activation_dropout <= 1.0:
            raise ValueError(f"activation_dropout must be in [0, 1], got {self.activation_dropout}")
        
        if not 0.0 <= self.attn_dropout <= 1.0:
            raise ValueError(f"attn_dropout must be in [0, 1], got {self.attn_dropout}")
        
        # Optimization validation
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        
        if self.descriptor_lr is not None and self.descriptor_lr <= 0:
            raise ValueError(f"descriptor_lr must be > 0, got {self.descriptor_lr}")
        
        if self.task_lr is not None and self.task_lr <= 0:
            raise ValueError(f"task_lr must be > 0, got {self.task_lr}")
        
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be > 0, got {self.grad_clip}")
        
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        
        # Loss weight validation
        if self.atom_weight < 0:
            raise ValueError(f"atom_weight must be >= 0, got {self.atom_weight}")
        
        if self.coord_weight < 0:
            raise ValueError(f"coord_weight must be >= 0, got {self.coord_weight}")
        
        if self.charge_weight < 0:
            raise ValueError(f"charge_weight must be >= 0, got {self.charge_weight}")

        if self.lidi_node_weight < 0:
            raise ValueError(f"lidi_node_weight must be >= 0, got {self.lidi_node_weight}")

        if self.lidi_edge_weight < 0:
            raise ValueError(f"lidi_edge_weight must be >= 0, got {self.lidi_edge_weight}")

        if self.lidi_conservation_weight < 0:
            raise ValueError(
                f"lidi_conservation_weight must be >= 0, got {self.lidi_conservation_weight}"
            )

        valid_lidi_reduce = {"global", "per_molecule"}
        if self.lidi_reduce not in valid_lidi_reduce:
            raise ValueError(
                f"lidi_reduce must be one of {valid_lidi_reduce}, got {self.lidi_reduce!r}"
            )

        valid_loss_balance = {"fixed", "ema"}
        if self.loss_balance_mode not in valid_loss_balance:
            raise ValueError(
                f"loss_balance_mode must be one of {valid_loss_balance}, got {self.loss_balance_mode!r}"
            )

        if not 0.0 <= self.loss_ema_beta < 1.0:
            raise ValueError(f"loss_ema_beta must be in [0, 1), got {self.loss_ema_beta}")

        if self.loss_balance_eps <= 0:
            raise ValueError(f"loss_balance_eps must be > 0, got {self.loss_balance_eps}")

        valid_edge_transform = {"none", "asinh"}
        if self.lidi_edge_transform not in valid_edge_transform:
            raise ValueError(
                f"lidi_edge_transform must be one of {valid_edge_transform}, got {self.lidi_edge_transform!r}"
            )

        if self.lidi_edge_scale <= 0:
            raise ValueError(f"lidi_edge_scale must be > 0, got {self.lidi_edge_scale}")

        if self.lidi_hidden_dim <= 0:
            raise ValueError(f"lidi_hidden_dim must be > 0, got {self.lidi_hidden_dim}")

        valid_geometric_subtasks = {"atom_mask", "coord_denoise", "charge", "lidi"}
        unknown_subtasks = sorted(set(self.geometric_subtasks) - valid_geometric_subtasks)
        if unknown_subtasks:
            raise ValueError(
                f"Unknown geometric_subtasks: {unknown_subtasks}. "
                f"Valid values: {sorted(valid_geometric_subtasks)}"
            )
        
        if self.vip_vea_weight < 0:
            raise ValueError(f"vip_vea_weight must be >= 0, got {self.vip_vea_weight}")
        
        if self.fukui_weight < 0:
            raise ValueError(f"fukui_weight must be >= 0, got {self.fukui_weight}")
        
        # Training mode validation
        valid_modes = {"geometric_structure", "electronic_structure", "cdft"}
        if self.train_mode not in valid_modes:
            raise ValueError(
                f"train_mode must be one of {valid_modes}, got {self.train_mode!r}"
            )

        if self.train_mode == "geometric_structure" and len(self.geometric_subtasks) == 0:
            raise ValueError("geometric_subtasks must contain at least one task for geometric training.")
        
        # LR scheduler validation
        valid_schedulers = {"cosine", "linear", "none"}
        if self.lr_scheduler not in valid_schedulers:
            raise ValueError(
                f"lr_scheduler must be one of {valid_schedulers}, got {self.lr_scheduler!r}"
            )
        
        # Runtime validation
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        
        if self.save_every <= 0:
            raise ValueError(f"save_every must be > 0, got {self.save_every}")
        if self.log_interval < 0:
            raise ValueError(f"log_interval must be >= 0, got {self.log_interval}")
        
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
