"""Import shim for the vendored xyz2mol_tmc module.

The vendored `xyz2mol_tmc.py` uses the absolute import path
`xyz2mol_tm.huckel_to_smiles.xyz2mol_local`, which assumes it was installed
as a pip package with a `huckel_to_smiles/` subdirectory.  The vendored copy
stores `xyz2mol_local.py` at the top level of `steamroll/xyz2mol_tm/`
instead.  This module patches :data:`sys.modules` to bridge the gap so that
neither vendored file needs to be modified.

It also patches `atomic_valence` in `xyz2mol_local` to add valences for
lanthanides and actinides (e.g. Ce, Z=58) that are absent from the vendored
tables, enabling obabel-based connectivity for those elements.

Exports:
    TRANSITION_METALS_NUM: list of atomic numbers for transition metals
    get_tmc_mol: function to build an RDKit Mol for a transition-metal complex
    xyz2ac_obabel: function to build an adjacency matrix and proto-mol from coordinates
"""

import importlib.util
import sys
import types
from pathlib import Path

_TM_DIR = Path(__file__).parent / "xyz2mol_tm"

# Valence cap for lanthanides / actinides missing from the vendored table.
# Value of 20 matches the convention used for d-block metals in xyz2mol_local.
_EXTRA_VALENCES: dict[int, list[int]] = {
    58: [20],  # Ce
    59: [20],  # Pr
    60: [20],  # Nd
    62: [20],  # Sm
    63: [20],  # Eu
    64: [20],  # Gd
    65: [20],  # Tb
    66: [20],  # Dy
    67: [20],  # Ho
    68: [20],  # Er
    69: [20],  # Tm
    70: [20],  # Yb
    89: [20],  # Ac
    90: [20],  # Th
    91: [20],  # Pa
    92: [20],  # U
    93: [20],  # Np
    94: [20],  # Pu
    95: [20],  # Am
    96: [20],  # Cm
    97: [20],  # Bk
    98: [20],  # Cf
    99: [20],  # Es
    100: [20],  # Fm
    101: [20],  # Md
    102: [20],  # No
    103: [20],  # Lr
}


def _setup_virtual_packages() -> None:
    """Register xyz2mol_tm virtual packages in sys.modules.

    Creates stub package entries for `xyz2mol_tm` and
    `xyz2mol_tm.huckel_to_smiles`, then loads `xyz2mol_local.py` under
    the `xyz2mol_tm.huckel_to_smiles.xyz2mol_local` key so that the absolute
    import in `xyz2mol_tmc.py` resolves without any changes to vendored files.
    Patches missing valences into the loaded module's `atomic_valence` dict.
    """
    if "xyz2mol_tm.huckel_to_smiles.xyz2mol_local" in sys.modules:
        return

    if "xyz2mol_tm" not in sys.modules:
        pkg = types.ModuleType("xyz2mol_tm")
        pkg.__path__ = [str(_TM_DIR)]
        pkg.__package__ = "xyz2mol_tm"
        sys.modules["xyz2mol_tm"] = pkg

    if "xyz2mol_tm.huckel_to_smiles" not in sys.modules:
        sub = types.ModuleType("xyz2mol_tm.huckel_to_smiles")
        sub.__path__ = [str(_TM_DIR)]
        sub.__package__ = "xyz2mol_tm.huckel_to_smiles"
        sys.modules["xyz2mol_tm.huckel_to_smiles"] = sub

    spec = importlib.util.spec_from_file_location(
        "xyz2mol_tm.huckel_to_smiles.xyz2mol_local",
        _TM_DIR / "xyz2mol_local.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load xyz2mol_local.py from {_TM_DIR}")
    local_mod = importlib.util.module_from_spec(spec)
    sys.modules["xyz2mol_tm.huckel_to_smiles.xyz2mol_local"] = local_mod
    spec.loader.exec_module(local_mod)  # type: ignore[union-attr]

    # Patch missing valences so xyz2AC_obabel handles lanthanides/actinides.
    for atomic_num, valences in _EXTRA_VALENCES.items():
        local_mod.atomic_valence[atomic_num] = valences  # type: ignore[union-attr]


_setup_virtual_packages()

_spec = importlib.util.spec_from_file_location(
    "steamroll.xyz2mol_tm.xyz2mol_tmc",
    _TM_DIR / "xyz2mol_tmc.py",
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load xyz2mol_tmc.py from {_TM_DIR}")
_tmc_mod = importlib.util.module_from_spec(_spec)
sys.modules["steamroll.xyz2mol_tm.xyz2mol_tmc"] = _tmc_mod
_spec.loader.exec_module(_tmc_mod)  # type: ignore[union-attr]

TRANSITION_METALS_NUM: list[int] = _tmc_mod.TRANSITION_METALS_NUM
get_tmc_mol = _tmc_mod.get_tmc_mol
xyz2ac_obabel = sys.modules["xyz2mol_tm.huckel_to_smiles.xyz2mol_local"].xyz2AC_obabel
