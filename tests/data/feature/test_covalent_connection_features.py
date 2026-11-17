"""Explicit intra-chain bonds must reach guidance without duplicated forces."""

from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable
import torch

from boltz.data import const
from boltz.data.feature.featurizerv2 import process_chain_feature_constraints
from boltz.data.types import (
    BondV2,
    Chain,
)
from boltz.model.potentials.potentials import ConnectionsPotential

COVALENT = const.bond_type_ids["COVALENT"]
SINGLE = const.bond_type_ids["SINGLE"]


@pytest.mark.parametrize(
    "rows,expected_atoms,expected_chains",
    [
        ([], [], []),
        ([(0, 0, 0, 0, 0, 1, SINGLE)], [], []),
        ([(0, 0, 0, 1, 2, 8, COVALENT)], [[2, 8]], []),
        ([(0, 0, 0, 0, 0, 2, COVALENT)], [[0, 2]], []),
        ([(0, 1, 0, 1, 2, 8, COVALENT)], [[2, 8]], [[0, 1]]),
        ([(0, 1, 0, 1, 2, 8, SINGLE)], [[2, 8]], [[0, 1]]),
        (
            [
                (0, 0, 0, 0, 0, 1, SINGLE),
                (0, 0, 0, 1, 2, 8, COVALENT),
                (0, 1, 1, 2, 8, 9, COVALENT),
            ],
            [[2, 8], [8, 9]],
            [[0, 1]],
        ),
    ],
)
def test_explicit_atom_and_chain_connectivity_are_separate(
    rows, expected_atoms, expected_chains
):
    chains = np.zeros(2, dtype=Chain)
    chains["entity_id"] = [0, 1]
    data = SimpleNamespace(
        structure=SimpleNamespace(chains=chains, bonds=np.array(rows, dtype=BondV2))
    )
    feats = process_chain_feature_constraints(data)
    assert feats["connected_atom_index"].T.tolist() == expected_atoms
    assert feats["connected_chain_index"].T.tolist() == expected_chains
    assert feats["connected_atom_index"].shape == (2, len(expected_atoms))
    assert feats["connected_chain_index"].shape == (2, len(expected_chains))


@pytest.mark.parametrize("distance", [1.8, 3.0, 8.0])
def test_same_chain_connection_energy_and_gradient(distance):
    chains = np.zeros(1, dtype=Chain)
    data = SimpleNamespace(
        structure=SimpleNamespace(
            chains=chains, bonds=np.array([(0, 0, 0, 1, 0, 1, COVALENT)], dtype=BondV2)
        )
    )
    feats = {k: v[None] for k, v in process_chain_feature_constraints(data).items()}
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]], requires_grad=True)
    potential = ConnectionsPotential()
    parameters = {"buffer": 2.0}
    energy = potential.compute(xyz, feats, parameters)
    assert float(energy.item()) == pytest.approx(max(distance - 2.0, 0.0))
    expected = torch.autograd.grad(energy.sum(), xyz)[0]
    actual = potential.compute_gradient(xyz.detach(), feats, parameters)
    torch.testing.assert_close(actual, expected)
    if distance > 2.0:
        after = potential.compute(xyz.detach() - 0.1 * actual, feats, parameters)
        assert (after < energy).all()


def test_repeated_and_reversed_internal_links_have_one_force():
    chains = np.zeros(1, dtype=Chain)
    rows = [
        (0, 0, 0, 1, 2, 8, COVALENT),
        (0, 0, 1, 0, 8, 2, COVALENT),
        (0, 0, 0, 1, 2, 8, COVALENT),
    ]
    data = SimpleNamespace(
        structure=SimpleNamespace(chains=chains, bonds=np.array(rows, dtype=BondV2))
    )
    before = data.structure.bonds.copy()
    features = process_chain_feature_constraints(data)
    assert features["connected_atom_index"].T.tolist() == [[2, 8]]
    np.testing.assert_array_equal(data.structure.bonds, before)


def test_self_bond_has_clear_error():
    data = SimpleNamespace(
        structure=SimpleNamespace(
            chains=np.zeros(1, dtype=Chain),
            bonds=np.array([(0, 0, 0, 0, 2, 2, COVALENT)], dtype=BondV2),
        )
    )
    with pytest.raises(ValueError, match="two distinct atoms"):
        process_chain_feature_constraints(data)


@pytest.mark.parametrize("elements", [(6, 16), (16, 16), (6, 6), (6, 7)])
def test_element_based_bounds_for_internal_links_only(elements):
    chains = np.zeros(2, dtype=Chain)
    data = SimpleNamespace(
        structure=SimpleNamespace(
            chains=chains,
            bonds=np.array(
                [
                    (0, 0, 0, 1, 0, 1, COVALENT),
                    (0, 1, 0, 2, 0, 2, COVALENT),
                ],
                dtype=BondV2,
            ),
        )
    )
    ref_elements = torch.nn.functional.one_hot(
        torch.tensor([*elements, 6]), const.num_elements
    )
    data.structure.atoms = np.zeros(3)
    data.tokens = np.array([(0, 3)], dtype=[("atom_idx", "i4"), ("atom_num", "i4")])
    feats = process_chain_feature_constraints(data, ref_elements)
    radius = sum(GetPeriodicTable().GetRcovalent(e) for e in elements)
    torch.testing.assert_close(
        feats["connected_atom_lower_bounds"], torch.tensor([radius * 0.875, 0.0])
    )
    torch.testing.assert_close(
        feats["connected_atom_upper_bounds"], torch.tensor([radius * 1.125, 0.0])
    )


def test_unknown_element_has_explicit_fallback():
    data = SimpleNamespace(
        structure=SimpleNamespace(
            chains=np.zeros(1, dtype=Chain),
            bonds=np.array([(0, 0, 0, 1, 0, 1, COVALENT)], dtype=BondV2),
        )
    )
    elements = torch.nn.functional.one_hot(torch.tensor([0, 16]), const.num_elements)
    data.structure.atoms = np.zeros(2)
    data.tokens = np.array([(0, 2)], dtype=[("atom_idx", "i4"), ("atom_num", "i4")])
    with pytest.warns(UserWarning, match="legacy connection upper bound"):
        feats = process_chain_feature_constraints(data, elements)
    assert feats["connected_atom_lower_bounds"].tolist() == [0.0]
    assert feats["connected_atom_upper_bounds"].tolist() == [0.0]


@pytest.mark.parametrize("distance", [0.0, 1e-10, 1.2, 1.8, 2.5])
def test_bounded_connection_energy_and_gradient(distance):
    features = {
        "connected_atom_index": torch.tensor([[[0], [1]]]),
        "connected_atom_lower_bounds": torch.tensor([[1.58375]]),
        "connected_atom_upper_bounds": torch.tensor([[2.03625]]),
    }
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]], requires_grad=True)
    potential = ConnectionsPotential()
    energy = potential.compute(xyz, features, {"buffer": 2.0})
    actual = potential.compute_gradient(xyz.detach(), features, {"buffer": 2.0})
    expected = torch.autograd.grad(energy.sum(), xyz)[0]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    if 0 < distance < 1.58375:
        assert actual[0, 1, 0] < 0  # Gradient descent separates compressed endpoints.
    elif distance > 2.03625:
        assert actual[0, 1, 0] > 0


def _named_molecule(smiles, names):
    mol = Chem.MolFromSmiles(smiles)
    for atom, name in zip(mol.GetAtoms(), names):
        atom.SetProp("name", name)
        atom.SetProp("leaving_atom", "0")
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, (float(i), 0.2 * (i % 2), 0.0))
    mol.AddConformer(conformer)
    return mol


@pytest.mark.parametrize("crop", [False, True])
def test_yaml_to_featurizer_to_guidance_with_modified_residues(tmp_path, crop):
    """Exercise production feature assembly, including a retained cropped link."""
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.parse.schema import parse_boltz_schema
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Input
    from boltz.model.layers.structure_checks import check_structures
    from boltz.model.potentials.potentials import IntraChainGeometryPotential

    molecules = {
        "ACE": _named_molecule("C(=O)C", ["C", "O", "CH3"]),
        "CY3": _named_molecule("NCC(=O)CS", ["N", "CA", "C", "O", "CB", "SG"]),
        "CAR": _named_molecule("C", ["C1"]),
    }
    schema = {
        "version": 1,
        "sequences": [
            {"ligand": {"id": "A", "ccd": "CAR"}},
            {
                "protein": {
                    "id": "B",
                    "sequence": "GC",
                    "msa": "empty",
                    "modifications": [
                        {"position": 1, "ccd": "ACE"},
                        {"position": 2, "ccd": "CY3"},
                    ],
                }
            },
        ],
        "constraints": [
            {"bond": {"atom1": ["B", 1, "CH3"], "atom2": ["B", 2, "SG"]}},
            {"bond": {"atom1": ["B", 2, "SG"], "atom2": ["B", 1, "CH3"]}},
        ],
    }
    target = parse_boltz_schema(
        Path("closure.yaml"), schema, molecules, tmp_path, boltz_2=True
    )
    data = Boltz2Tokenizer().tokenize(
        Input(
            structure=target.structure,
            record=target.record,
            msa={},
            residue_constraints=target.residue_constraints,
            extra_mols=target.extra_mols,
        )
    )
    if crop:
        data = replace(data, tokens=data.tokens[data.tokens["asym_id"] == 1])
    featurizer = Boltz2Featurizer()
    torch_rng = torch.get_rng_state()
    feats = featurizer.process(
        data,
        np.random.default_rng(7),
        molecules,
        training=False,
        max_seqs=1,
        compute_constraint_features=True,
    )
    after = torch.get_rng_state()
    # Guidance metadata must not change any learned network features or RNG use.
    torch.set_rng_state(torch_rng)
    no_constraints = featurizer.process(
        data,
        np.random.default_rng(7),
        molecules,
        training=False,
        max_seqs=1,
        compute_constraint_features=False,
    )
    assert torch.equal(torch.get_rng_state(), after)
    for key, value in no_constraints.items():
        torch.testing.assert_close(
            feats[key], value, rtol=0, atol=0, equal_nan=True, msg=key
        )

    assert feats["connected_atom_index"].shape == (2, 1)
    assert feats["connected_chain_index"].shape == (2, 0)
    endpoints = feats["connected_atom_index"][:, 0]
    assert feats["ref_element"][endpoints].argmax(-1).tolist() == [6, 16]
    torch.testing.assert_close(
        feats["connected_atom_lower_bounds"], torch.tensor([1.58375])
    )
    torch.testing.assert_close(
        feats["connected_atom_upper_bounds"], torch.tensor([2.03625])
    )

    # The parser's explicit closure bound remains for the fork's output checks,
    # but it must not also be part of the added geometry contribution.
    pairs = feats["rdkit_bounds_index"]
    closure = (pairs.sort(dim=0).values == endpoints.sort().values[:, None]).all(dim=0)
    assert closure.sum() == 1
    selected = feats["intrachain_covalent_bound_indices"]
    assert not closure[selected].any()
    assert selected.numel() > 0
    assert pairs.max() < int(feats["atom_pad_mask"].sum())
    batched = {k: v[None] for k, v in feats.items()}
    xyz = torch.randn(1, feats["ref_element"].shape[0], 3)
    for potential, params in (
        (ConnectionsPotential(), {"buffer": 2.0}),
        (
            IntraChainGeometryPotential(),
            {"bond_buffer": 0.125, "angle_buffer": 0.125, "clash_buffer": 0.1},
        ),
    ):
        assert torch.isfinite(potential.compute(xyz, batched, params)).all()
        assert torch.isfinite(potential.compute_gradient(xyz, batched, params)).all()
    checks = check_structures(xyz, batched, multiplicity=1)
    assert int(checks["bonds_checked"][0]) == int(
        (
            feats["rdkit_bounds_bond_mask"]
            & ~feats["rdkit_bounds_angle_mask"]
            & ~feats["rdkit_bounds_guidance_only_mask"]
        ).sum()
    )
    assert feats["rdkit_bounds_guidance_only_mask"].sum() == 1  # Adjacent C--N prior.
