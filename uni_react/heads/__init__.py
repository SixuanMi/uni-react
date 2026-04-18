"""Task-head implementations.

All atomic heads are registered in :data:`~uni_react.registry.HEAD_REGISTRY`.
Pipeline tasks (bundles of multiple heads) live in this package too.

Atom-level heads
----------------
``atom_mask``       :class:`AtomMaskHead`
``coord_denoise``   :class:`CoordDenoiseHead`
``charge``          :class:`ChargeHead`
``lidi``            :class:`LidiHead`
``vip_vea``         :class:`VipVeaHead`
``fukui``           :class:`FukuiHead`

Pipeline bundles
----------------
:class:`GeometricStructureTask`   – atom_mask + coord_denoise + charge
:class:`ElectronicStructureTask`  – vip_vea + fukui
"""
from .atom_mask import AtomMaskHead
from .charge import ChargeHead
from .coord_denoise import CoordDenoiseHead
from .electronic_pipeline import ElectronicStructureTask
from .fukui import FukuiHead
from .geometric_pipeline import GeometricStructureTask
from .lidi import LidiHead
from .vip_vea import VipVeaHead

__all__ = [
    # atomic heads
    "AtomMaskHead",
    "CoordDenoiseHead",
    "ChargeHead",
    "LidiHead",
    "VipVeaHead",
    "FukuiHead",
    # pipeline bundles
    "GeometricStructureTask",
    "ElectronicStructureTask",
]
