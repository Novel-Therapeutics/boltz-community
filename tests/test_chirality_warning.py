"""The predict-time check for chiral modified residues in polymer chains.

Chiral atom constraints are only applied with --use_potentials. Without it,
D-amino acids in a protein chain came out inverted or flattened in 2 of 3
test predictions (reported upstream by @TSC2412 in jwohlwend/boltz#710).
"""

from types import SimpleNamespace

import numpy as np

from boltz.data.types import ChiralAtomConstraint, ResidueConstraints
from boltz.main import BoltzProcessedInput, _records_with_polymer_chiral_centers


def _constraints(centers):
    chiral = np.zeros(len(centers), dtype=ChiralAtomConstraint)
    for i, center in enumerate(centers):
        chiral[i]["atom_idxs"] = (center, center, center, center)
        chiral[i]["is_reference"] = True
        chiral[i]["is_r"] = True
    empty = np.zeros(0, dtype=[("x", "i4")])
    return ResidueConstraints(
        rdkit_bounds_constraints=empty,
        chiral_atom_constraints=chiral,
        stereo_bond_constraints=empty,
        planar_bond_constraints=empty,
        planar_ring_5_constraints=empty,
        planar_ring_6_constraints=empty,
    )


def _processed(tmp_path, structure, centers):
    targets = tmp_path / "structures"
    constraints = tmp_path / "constraints"
    targets.mkdir()
    constraints.mkdir()
    structure.dump(targets / "rec.npz")
    if centers is not None:
        _constraints(centers).dump(constraints / "rec.npz")
    return BoltzProcessedInput(
        manifest=SimpleNamespace(records=[SimpleNamespace(id="rec")]),
        targets_dir=targets,
        msa_dir=tmp_path / "msa",
        constraints_dir=constraints,
    )


def test_chiral_center_in_protein_chain_is_flagged(tmp_path, minimal_structure_with_ligand):
    processed = _processed(tmp_path, minimal_structure_with_ligand, centers=[1])
    assert _records_with_polymer_chiral_centers(processed, boltz2=False) == ["rec"]


def test_chiral_center_in_ligand_is_not_flagged(tmp_path, minimal_structure_with_ligand):
    ligand_atom = int(minimal_structure_with_ligand.chains[1]["atom_idx"])
    processed = _processed(tmp_path, minimal_structure_with_ligand, centers=[ligand_atom])
    assert _records_with_polymer_chiral_centers(processed, boltz2=False) == []


def test_no_chiral_constraints_is_not_flagged(tmp_path, minimal_structure_with_ligand):
    processed = _processed(tmp_path, minimal_structure_with_ligand, centers=[])
    assert _records_with_polymer_chiral_centers(processed, boltz2=False) == []


def test_missing_constraints_file_is_not_flagged(tmp_path, minimal_structure_with_ligand):
    processed = _processed(tmp_path, minimal_structure_with_ligand, centers=None)
    assert _records_with_polymer_chiral_centers(processed, boltz2=False) == []
