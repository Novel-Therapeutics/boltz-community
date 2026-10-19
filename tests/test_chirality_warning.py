"""The predict-time warning for stereocentres when --use_potentials is off.

Chiral atom constraints are only applied with --use_potentials. Without it,
Boltz-2 often inverted or flattened stereocentres: D-amino acids in a protein
chain (reported upstream by @TSC2412 in jwohlwend/boltz#710) and ligand
stereocentres alike, in a protein as well as on their own.
"""

from types import SimpleNamespace

import numpy as np

from boltz.data.types import ChiralAtomConstraint, ResidueConstraints
from boltz.main import BoltzProcessedInput, _records_with_stereocenters


def _constraints(n_centers):
    chiral = np.zeros(n_centers, dtype=ChiralAtomConstraint)
    for i in range(n_centers):
        chiral[i]["atom_idxs"] = (0, 1, 2, 3)
        chiral[i]["is_reference"] = True
    empty = np.zeros(0, dtype=[("x", "i4")])
    return ResidueConstraints(
        rdkit_bounds_constraints=empty,
        chiral_atom_constraints=chiral,
        stereo_bond_constraints=empty,
        planar_bond_constraints=empty,
        planar_ring_5_constraints=empty,
        planar_ring_6_constraints=empty,
    )


def _processed(tmp_path, records):
    """records maps record id -> number of stereocentres (None = no file)."""
    constraints = tmp_path / "constraints"
    constraints.mkdir()
    for rec, n in records.items():
        if n is not None:
            _constraints(n).dump(constraints / f"{rec}.npz")
    return BoltzProcessedInput(
        manifest=SimpleNamespace(records=[SimpleNamespace(id=r) for r in records]),
        targets_dir=tmp_path / "structures",
        msa_dir=tmp_path / "msa",
        constraints_dir=constraints,
    )


def test_records_with_stereocentres_are_flagged(tmp_path):
    processed = _processed(tmp_path, {"chiral": 3, "achiral": 0, "no_file": None})
    assert _records_with_stereocenters(processed) == ["chiral"]


def test_no_constraints_dir_flags_nothing(tmp_path):
    processed = BoltzProcessedInput(
        manifest=SimpleNamespace(records=[SimpleNamespace(id="x")]),
        targets_dir=tmp_path,
        msa_dir=tmp_path,
        constraints_dir=None,
    )
    assert _records_with_stereocenters(processed) == []
