"""SingleMolPretrainNet: encoder backbone + task-head composition."""
from typing import Dict, Iterable, Optional

import torch

from .single_mol import SingleMolEncoder
from ..heads import ElectronicStructureTask, GeometricStructureTask


class SingleMolPretrainNet(torch.nn.Module):
    """Pretraining model: SingleMolEncoder backbone + geometric/electronic task heads.

    The ``descriptor`` attribute holds the encoder so that existing optimizer
    code (which splits LR by ``descriptor.*`` vs everything else) keeps working.

    Checkpoint compatibility
    ------------------------
    ``load_state_dict`` automatically remaps legacy key prefixes from older
    checkpoint formats to the current nested layout.
    """

    _LEGACY_BACKBONE_PREFIXES = (
        "inv_mpnns.", "message_layers.", "FTEs.", "SAs.",
        "lns_before_attn.", "lns_after_attn.", "lns_after_mp.",
        "atom_encoder.", "bond_encoder.",
        "dist_to_bond.", "src_to_bond.", "tgt_to_bond.", "bond_to_bias.",
        "Svec.", "lin.",
    )

    def __init__(
        self,
        emb_dim: int,
        inv_layer: int = 2,
        se3_layer: int = 4,
        heads: int = 8,
        atom_vocab_size: int = 128,
        cutoff: float = 5.0,
        path_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        num_kernel: int = 128,
        enable_electronic_structure_task: bool = False,
        electronic_structure_vip_vea_dim: int = 2,
        electronic_structure_fukui_dim: int = 3,
        enable_lidi_task: bool = False,
        geometric_lidi_hidden_dim: int = 256,
        encoder: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)

        # Accept a pre-built encoder or build the default SingleMolEncoder.
        # Named ``descriptor`` for backward-compat with optimizer LR splitting.
        if encoder is not None:
            self.descriptor = encoder
        else:
            self.descriptor = SingleMolEncoder(
                emb_dim=emb_dim, inv_layer=inv_layer, se3_layer=se3_layer,
                heads=heads, atom_vocab_size=atom_vocab_size, cutoff=cutoff,
                path_dropout=path_dropout, activation_dropout=activation_dropout,
                attn_dropout=attn_dropout, num_kernel=num_kernel,
            )

        self.tasks = torch.nn.ModuleDict({
            "geometric_structure": GeometricStructureTask(
                emb_dim=emb_dim, atom_vocab_size=atom_vocab_size,
                enable_lidi=enable_lidi_task,
                lidi_hidden_dim=geometric_lidi_hidden_dim,
            ),
        })
        if enable_electronic_structure_task:
            self.tasks["electronic_structure"] = ElectronicStructureTask(
                emb_dim=emb_dim,
                vip_vea_dim=electronic_structure_vip_vea_dim,
                fukui_dim=electronic_structure_fukui_dim,
            )
        self.default_pipeline_tasks = ("geometric_structure",)
        self.task_atom_heads  = torch.nn.ModuleDict()
        self.task_graph_heads = torch.nn.ModuleDict()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Backward-compat property accessors
    # ------------------------------------------------------------------
    @property
    def atom_head(self):
        return self.tasks["geometric_structure"].atom_mask.head

    @property
    def coord_head(self):
        return self.tasks["geometric_structure"].coord_denoise.head

    @property
    def charge_head(self):
        return self.tasks["geometric_structure"].charge.head

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def extract_descriptors(
        self,
        input_atomic_numbers: torch.Tensor,
        coords_noisy: torch.Tensor,
        atom_padding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.descriptor(
            input_atomic_numbers=input_atomic_numbers,
            coords_noisy=coords_noisy,
            atom_padding=atom_padding,
        )

    def forward(
        self,
        input_atomic_numbers: torch.Tensor,
        coords_noisy: torch.Tensor,
        atom_padding: Optional[torch.Tensor] = None,
        active_pipeline_tasks: Optional[Iterable[str]] = None,
        active_geometric_subtasks: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        descriptors = self.extract_descriptors(
            input_atomic_numbers=input_atomic_numbers,
            coords_noisy=coords_noisy,
            atom_padding=atom_padding,
        )
        pipeline_tasks = tuple(active_pipeline_tasks) if active_pipeline_tasks is not None \
            else self.default_pipeline_tasks
        out = dict(descriptors)
        for task_name in pipeline_tasks:
            if task_name not in self.tasks:
                raise KeyError(f"Unknown pipeline task: {task_name!r}")
            if task_name == "geometric_structure":
                out.update(
                    self.tasks[task_name](
                        descriptors,
                        active_subtasks=active_geometric_subtasks,
                    )
                )
            else:
                out.update(self.tasks[task_name](descriptors))
        for name, head in self.task_atom_heads.items():
            out[name] = head(descriptors["node_feats"])
        for name, head in self.task_graph_heads.items():
            out[name] = head(descriptors["graph_feats"])
        return out

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def compute_geometric_structure_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        atom_weight: float = 1.0,
        coord_weight: float = 1.0,
        charge_weight: float = 1.0,
        active_subtasks: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.tasks["geometric_structure"].compute_loss_dict(
            outputs=outputs, batch=batch,
            atom_weight=atom_weight, coord_weight=coord_weight,
            charge_weight=charge_weight, active_subtasks=active_subtasks,
        )

    def compute_electronic_structure_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        vip_vea_weight: float = 1.0,
        fukui_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if "electronic_structure" not in self.tasks:
            raise RuntimeError("Enable electronic_structure task first.")
        return self.tasks["electronic_structure"].compute_loss_dict(
            outputs=outputs, batch=batch,
            vip_vea_weight=vip_vea_weight, fukui_weight=fukui_weight,
        )

    # ------------------------------------------------------------------
    # Legacy checkpoint key remapping
    # ------------------------------------------------------------------

    @classmethod
    def _remap_legacy_key(cls, key: str) -> str:
        if key.startswith("descriptor.Svec.proj.0."):
            return "descriptor.Svec.net.0." + key[len("descriptor.Svec.proj.0."):]
        if key.startswith("descriptor.Svec.proj.2."):
            return "descriptor.Svec.net.3." + key[len("descriptor.Svec.proj.2."):]
        if key.startswith("Svec.proj.0."):
            return "descriptor.Svec.net.0." + key[len("Svec.proj.0."):]
        if key.startswith("Svec.proj.2."):
            return "descriptor.Svec.net.3." + key[len("Svec.proj.2."):]
        if key.startswith("tasks.atom_mask."):
            return "tasks.geometric_structure.atom_mask." + key[len("tasks.atom_mask."):]
        if key.startswith("tasks.coord_denoise."):
            return "tasks.geometric_structure.coord_denoise." + key[len("tasks.coord_denoise."):]
        if key.startswith("tasks.charge."):
            return "tasks.geometric_structure.charge." + key[len("tasks.charge."):]
        if key.startswith("tasks.reactivity."):
            return "tasks.electronic_structure." + key[len("tasks.reactivity."):]
        if key.startswith(("descriptor.", "tasks.", "task_atom_heads.", "task_graph_heads.")):
            return key
        if key.startswith("atom_head."):
            return "tasks.geometric_structure.atom_mask.head." + key[len("atom_head."):]
        if key.startswith("coord_head."):
            return "tasks.geometric_structure.coord_denoise.head." + key[len("coord_head."):]
        if key.startswith("charge_head."):
            return "tasks.geometric_structure.charge.head." + key[len("charge_head."):]
        for prefix in cls._LEGACY_BACKBONE_PREFIXES:
            if key.startswith(prefix):
                return "descriptor." + key
        return key

    def load_state_dict(self, state_dict, strict: bool = True):
        remapped = {}
        for key, value in state_dict.items():
            new_key = self._remap_legacy_key(key)
            if new_key in remapped and new_key != key:
                continue
            remapped[new_key] = value
        return super().load_state_dict(remapped, strict=strict)
