"""Stereochemistry and bond-geometry checks on predicted structures."""

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from boltz.data.parse.schema import (
    compute_chiral_atom_constraints,
    compute_geometry_constraints,
)
from boltz.model.layers.structure_checks import (
    FLAT_DIHEDRAL,
    check_structures,
    dihedral,
    passes_checks,
    select_affinity_sample,
)
from boltz.model.potentials.potentials import ChiralAtomPotential

DEXAMETHASONE = (
    "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@@]4([C@]3([C@H](C[C@@]2"
    "([C@]1(C(=O)CO)O)C)O)F)C"
)
GLUCOSE = "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O"


def _ligand(smiles):
    """Ideal heavy-atom coordinates and the parser's constraints as features."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=11)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    idx_map = {h: i for i, h in enumerate(heavy)}
    chiral = compute_chiral_atom_constraints(mol, idx_map)
    bounds = compute_geometry_constraints(mol, idx_map)
    feats = {
        "chiral_atom_index": torch.tensor([c.atom_idxs for c in chiral]).T[None],
        "chiral_reference_mask": torch.tensor([c.is_reference for c in chiral])[None],
        "chiral_atom_orientations": torch.tensor([c.is_r for c in chiral])[None],
        "rdkit_bounds_index": torch.tensor([c.atom_idxs for c in bounds]).T[None],
        "rdkit_bounds_bond_mask": torch.tensor([c.is_bond for c in bounds])[None],
        "rdkit_bounds_angle_mask": torch.tensor([c.is_angle for c in bounds])[None],
        "rdkit_lower_bounds": torch.tensor([c.lower_bound for c in bounds])[None],
        "rdkit_upper_bounds": torch.tensor([c.upper_bound for c in bounds])[None],
    }
    coords = torch.tensor(mol.GetConformer().GetPositions()[heavy], dtype=torch.float)
    return coords, feats, mol, heavy


def test_dihedral_matches_the_chirality_potential():
    """Same convention as the potential that --use_potentials applies."""
    torch.manual_seed(0)
    coords = torch.randn(10, 3)
    index = torch.stack([torch.randperm(10)[:4] for _ in range(6)], dim=1)
    expected = ChiralAtomPotential(parameters={}).compute_variable(coords, index)
    assert torch.allclose(dihedral(coords, index), expected, atol=1e-6)


@pytest.mark.parametrize("smiles,n_centers", [(GLUCOSE, 5), (DEXAMETHASONE, 8)])
def test_ideal_geometry_passes(smiles, n_centers):
    coords, feats, _, _ = _ligand(smiles)
    checks = check_structures(coords[None], feats, multiplicity=1)
    assert checks["chiral_centers_checked"].item() == n_centers
    assert checks["chiral_centers_inverted"].item() == 0
    assert checks["chiral_centers_flattened"].item() == 0
    assert checks["bonds_checked"].item() > 0
    assert checks["bonds_out_of_range"].item() == 0
    assert passes_checks(checks).item()


@pytest.mark.parametrize("smiles,n_centers", [(GLUCOSE, 5), (DEXAMETHASONE, 8)])
def test_mirror_image_inverts_every_center(smiles, n_centers):
    coords, feats, _, _ = _ligand(smiles)
    mirrored = coords * torch.tensor([-1.0, 1.0, 1.0])
    checks = check_structures(mirrored[None], feats, multiplicity=1)
    assert checks["chiral_centers_inverted"].item() == n_centers
    assert checks["bonds_out_of_range"].item() == 0  # mirroring keeps every bond length
    assert not passes_checks(checks).item()


def test_flattened_center_is_caught():
    coords, feats, mol, heavy = _ligand(GLUCOSE)
    ref = feats["chiral_reference_mask"][0]
    n1, n2, n3, center = feats["chiral_atom_index"][0][:, ref][:, 0].tolist()
    # Move the centre into the plane of its three neighbours.
    a, b, c = coords[n1], coords[n2], coords[n3]
    normal = torch.cross(b - a, c - a, dim=-1)
    normal = normal / normal.norm()
    flat = coords.clone()
    flat[center] = coords[center] - ((coords[center] - a) @ normal) * normal
    checks = check_structures(flat[None], feats, multiplicity=1)
    assert checks["chiral_centers_flattened"].item() >= 1


def test_torn_bond_is_caught():
    coords, feats, _, _ = _ligand(DEXAMETHASONE)
    torn = coords.clone()
    torn[0] += torch.tensor([2.0, 0.0, 0.0])  # pull a methyl carbon off its ring
    checks = check_structures(torn[None], feats, multiplicity=1)
    assert checks["bonds_out_of_range"].item() >= 1


def test_samples_are_checked_independently():
    coords, feats, _, _ = _ligand(GLUCOSE)
    both = torch.stack([coords, coords * torch.tensor([-1.0, 1.0, 1.0])])
    checks = check_structures(both, feats, multiplicity=2)
    assert checks["chiral_centers_inverted"].tolist() == [0, 5]
    assert passes_checks(checks).tolist() == [True, False]


def test_no_constraints_reports_zero_checked():
    coords = torch.zeros(1, 4, 3)
    empty = {
        "chiral_atom_index": torch.zeros(1, 4, 0, dtype=torch.long),
        "chiral_reference_mask": torch.zeros(1, 0, dtype=torch.bool),
        "chiral_atom_orientations": torch.zeros(1, 0, dtype=torch.bool),
    }
    checks = check_structures(coords, empty, multiplicity=1)
    assert checks["chiral_centers_checked"].item() == 0
    assert passes_checks(checks).item()


def test_affinity_sample_prefers_passing_samples():
    iptm = torch.tensor([0.9, 0.8, 0.7])
    assert select_affinity_sample(iptm, torch.tensor([False, True, True])) == 1
    assert select_affinity_sample(iptm, torch.tensor([True, True, True])) == 0
    # None pass: fall back to the highest ipTM, as before.
    assert select_affinity_sample(iptm, torch.tensor([False, False, False])) == 0


def test_flat_threshold_is_well_below_ideal_centers():
    """Ideal centres in these molecules sit at about 29-40 degrees."""
    for smiles in (GLUCOSE, DEXAMETHASONE):
        coords, feats, _, _ = _ligand(smiles)
        ref = feats["chiral_reference_mask"][0]
        phi = dihedral(coords, feats["chiral_atom_index"][0][:, ref])
        assert (phi.abs() > 2 * FLAT_DIHEDRAL).all()
