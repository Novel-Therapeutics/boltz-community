"""Constraint features must follow the atoms through a crop.

Constraints are built with atom indices into the full structure, while the
atom features hold only the cropped tokens' atoms. The affinity pass crops to
the binding pocket, so once it is steered (--use_potentials) the potentials
read these indices against the cropped coordinates.
"""

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boltz.data.feature.featurizerv2 import remap_constraint_features

TOKEN_DTYPE = [("atom_idx", "i4"), ("atom_num", "i4"), ("asym_id", "i4")]

# Chain 0: four 2-atom residues (atoms 0-7). Chain 1: a 4-atom ligand, one
# token per atom (atoms 8-11). Chain 2: a copy of chain 0 (atoms 12-19).
ALL_TOKENS = (
    [(2 * i, 2, 0) for i in range(4)]
    + [(8 + i, 1, 1) for i in range(4)]
    + [(12 + 2 * i, 2, 2) for i in range(4)]
)
NUM_ATOMS = 20


def _data(tokens):
    return SimpleNamespace(
        structure=SimpleNamespace(atoms=np.zeros(NUM_ATOMS)),
        tokens=np.array(tokens, dtype=TOKEN_DTYPE),
    )


def _features():
    return {
        "chiral_atom_index": torch.tensor([[8, 0], [9, 1], [10, 2], [11, 3]]),
        "chiral_reference_mask": torch.tensor([True, True]),
        "chiral_atom_orientations": torch.tensor([True, False]),
        "rdkit_bounds_index": torch.tensor([[4, 0, 7], [8, 1, 12]]),
        "rdkit_bounds_bond_mask": torch.tensor([True, True, False]),
        "rdkit_bounds_angle_mask": torch.tensor([False, False, True]),
        "rdkit_upper_bounds": torch.tensor([1.0, 2.0, 3.0]),
        "rdkit_lower_bounds": torch.tensor([0.5, 1.5, 2.5]),
        "stereo_bond_index": torch.empty((4, 0), dtype=torch.long),
        "stereo_reference_mask": torch.empty((0,), dtype=torch.bool),
        "stereo_bond_orientations": torch.empty((0,), dtype=torch.bool),
        "planar_bond_index": torch.empty((6, 0), dtype=torch.long),
        "planar_ring_5_index": torch.empty((5, 0), dtype=torch.long),
        "planar_ring_6_index": torch.empty((6, 0), dtype=torch.long),
        "connected_atom_index": torch.tensor([[7, 8], [8, 12]]).T,
        "connected_chain_index": torch.tensor([[0, 1], [1, 2]]).T,
        "symmetric_chain_index": torch.tensor([[0], [2]]),
        "contact_pair_index": torch.tensor([[8, 8, 8, 9], [6, 7, 0, 7]]),
        "contact_union_index": torch.tensor([0, 0, 1, 2]),
        "contact_negation_mask": torch.tensor([True, True, True, False]),
        "contact_thresholds": torch.tensor([4.0, 4.0, 5.0, 6.0]),
    }


def test_uncropped_features_are_unchanged():
    features = _features()
    remapped = remap_constraint_features(_data(ALL_TOKENS), features)
    assert remapped.keys() == features.keys()
    for key, value in features.items():
        assert torch.equal(remapped[key], value), key


class TestCrop:
    """Keep residues 2-3 of chain 0 and the ligand; drop the rest.

    Structure atoms 4-11 become featurized atoms 0-7.
    """

    @pytest.fixture
    def remapped(self):
        return remap_constraint_features(
            _data(ALL_TOKENS[2:8]), _features()
        )

    def test_chirality_points_at_the_cropped_ligand(self, remapped):
        assert remapped["chiral_atom_index"].tolist() == [[4], [5], [6], [7]]
        assert remapped["chiral_atom_orientations"].tolist() == [True]
        assert remapped["chiral_reference_mask"].tolist() == [True]

    def test_bounds_on_removed_atoms_are_dropped(self, remapped):
        assert remapped["rdkit_bounds_index"].tolist() == [[0], [4]]
        assert remapped["rdkit_bounds_bond_mask"].tolist() == [True]
        assert remapped["rdkit_bounds_angle_mask"].tolist() == [False]
        assert remapped["rdkit_upper_bounds"].tolist() == [1.0]
        assert remapped["rdkit_lower_bounds"].tolist() == [0.5]

    def test_connections_follow_their_atoms(self, remapped):
        assert remapped["connected_atom_index"].tolist() == [[3], [4]]
        assert remapped["connected_chain_index"].tolist() == [[0], [1]]

    def test_chain_pairs_with_a_removed_chain_are_dropped(self, remapped):
        assert remapped["symmetric_chain_index"].shape == (2, 0)

    def test_contact_unions_are_renumbered(self, remapped):
        assert remapped["contact_pair_index"].tolist() == [[4, 4, 5], [2, 3, 3]]
        assert remapped["contact_union_index"].tolist() == [0, 0, 1]
        assert remapped["contact_negation_mask"].tolist() == [True, True, False]
        assert remapped["contact_thresholds"].tolist() == [4.0, 4.0, 6.0]

    def test_empty_constraints_stay_empty(self, remapped):
        assert remapped["stereo_bond_index"].shape == (4, 0)
        assert remapped["planar_ring_6_index"].shape == (6, 0)


MOL_DIR = Path("~/.boltz/mols").expanduser()


@pytest.mark.slow
@pytest.mark.skipif(not MOL_DIR.exists(), reason="needs the Boltz molecule cache")
def test_steering_potentials_run_on_a_cropped_affinity_input(tmp_path):
    """A 300-residue protein followed by a chiral ligand, through the real
    affinity data path, must give potentials indices inside the crop."""
    from rdkit import Chem

    from boltz.data.mol import load_canonicals
    from boltz.data.module.inferencev2 import PredictionDataset, collate
    from boltz.data.parse.schema import parse_boltz_schema
    from boltz.data.types import Manifest
    from boltz.main import BoltzSteeringParams
    from boltz.model.potentials.potentials import get_potentials

    schema = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": "ACDEFGHIKLMNPQRSTVWY" * 15,
                    "msa": "empty",
                }
            },
            {"ligand": {"id": "B", "smiles": "C[C@H](N)C(=O)O"}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }
    target = parse_boltz_schema(
        Path("crop.yaml"), schema, load_canonicals(MOL_DIR), MOL_DIR, boltz_2=True
    )
    structure = target.structure
    record = target.record

    # Lay residues along a line and put the ligand beside residue 150, so the
    # crop keeps a window from the middle of the protein.
    coords = np.zeros((len(structure.atoms), 3), dtype=np.float32)
    for res in structure.residues:
        start, end = res["atom_idx"], res["atom_idx"] + res["atom_num"]
        coords[start:end, 0] = 3.8 * res["res_idx"]
    ligand = structure.chains[1]
    lig_start = ligand["atom_idx"]
    lig_end = lig_start + ligand["atom_num"]
    coords[lig_start:lig_end] = [3.8 * 150, 4.0, 0.0]
    coords[lig_start:lig_end, 2] += np.arange(lig_end - lig_start)
    structure.atoms["coords"] = coords
    structure.atoms["is_present"] = True
    structure.coords["coords"] = coords

    target_dir = tmp_path / "predictions"
    (target_dir / record.id).mkdir(parents=True)
    structure.dump(target_dir / record.id / f"pre_affinity_{record.id}.npz")
    constraints_dir = tmp_path / "constraints"
    constraints_dir.mkdir()
    target.residue_constraints.dump(constraints_dir / f"{record.id}.npz")
    extra_mols_dir = tmp_path / "mols"
    extra_mols_dir.mkdir()
    # As main does, so the ligand keeps its atom names through the pickle.
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    with (extra_mols_dir / f"{record.id}.pkl").open("wb") as f:
        pickle.dump(target.extra_mols, f)

    dataset = PredictionDataset(
        manifest=Manifest(records=[record]),
        target_dir=target_dir,
        msa_dir=tmp_path / "msa",
        mol_dir=MOL_DIR,
        constraints_dir=constraints_dir,
        extra_mols_dir=extra_mols_dir,
        override_method="other",
        affinity=True,
    )
    item = dataset[0]
    feats = collate([item])

    num_atoms = feats["atom_pad_mask"].shape[1]
    # The crop must actually drop atoms, or this test proves nothing.
    assert num_atoms < len(structure.atoms)

    # The ligand's chirality constraint lands on the ligand's featurized atoms.
    chiral_index = feats["chiral_atom_index"][0]
    assert chiral_index.shape[1] > 0
    atom_token = feats["atom_to_token"][0].argmax(dim=-1)
    ligand_token = feats["affinity_token_mask"][0].bool()
    assert ligand_token[atom_token[chiral_index]].all()

    steering = BoltzSteeringParams()
    steering.fk_steering = True
    steering.physical_guidance_update = True
    steering.contact_guidance_update = False
    sample = torch.randn(1, num_atoms, 3)
    for potential in get_potentials(vars(steering), boltz2=True):
        parameters = potential.compute_parameters(0.5)
        energy = potential.compute(sample, feats, parameters)
        assert torch.isfinite(energy).all(), type(potential).__name__

    # The structure checks read the same features on the affinity pass.
    from boltz.model.layers.structure_checks import check_structures

    checks = check_structures(sample, feats, multiplicity=1)
    num_centres = int(feats["chiral_reference_mask"][0].sum())
    assert num_centres > 0
    assert checks["chiral_centers_checked"].tolist() == [num_centres]
