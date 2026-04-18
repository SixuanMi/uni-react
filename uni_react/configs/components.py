"""Sub-config dataclasses for pluggable components.

Each dataclass maps directly to a registry entry via its ``type`` field.
Pass the dataclass as ``cfg.encoder``, ``cfg.loss``, etc. when building
components via the registry::

    from uni_react.registry import ENCODER_REGISTRY
    encoder = ENCODER_REGISTRY.build(dataclasses.asdict(cfg.encoder))
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EncoderConfig:
    """Configuration for the backbone encoder."""
    type: str = "single_mol"
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


@dataclass
class GeometricLossConfig:
    """Configuration for geometric-structure pretraining loss."""
    type: str = "geometric_structure"
    atom_weight: float = 1.0
    coord_weight: float = 1.0
    charge_weight: float = 1.0
    lidi_node_weight: float = 0.0
    lidi_edge_weight: float = 0.0
    lidi_conservation_weight: float = 0.0


@dataclass
class ElectronicLossConfig:
    """Configuration for electronic-structure pretraining loss."""
    type: str = "electronic_structure"
    vip_vea_weight: float = 1.0
    fukui_weight: float = 1.0
    vip_vea_keys: List[str] = field(default_factory=lambda: ["vip", "vea"])
    fukui_keys: List[str] = field(default_factory=lambda: ["f_plus", "f_minus", "f_zero"])


@dataclass
class LoggerConfig:
    """Configuration for the logging backend."""
    type: str = "console"
    # wandb-specific (ignored by console / tensorboard)
    project: Optional[str] = None
    name: Optional[str] = None
    entity: Optional[str] = None
    # tensorboard-specific
    log_dir: Optional[str] = None


@dataclass
class SchedulerConfig:
    """Configuration for the LR scheduler."""
    type: str = "cosine"   # cosine | linear | none
    warmup_steps: int = 1000
    total_steps: int = 0   # 0 = auto-computed from epochs * steps_per_epoch
    min_lr_ratio: float = 0.0
