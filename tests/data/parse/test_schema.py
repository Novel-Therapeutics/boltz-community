"""Tests for boltz.data.parse.schema — atom naming, chirality, and leaving atoms."""

import re

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from boltz.data.types import StructureV2


def _parse_ligand_smiles(smiles: str) -> StructureV2:
    """Run a SMILES string through the full production parse path."""
    from boltz.data.parse.schema import parse_boltz_schema

    schema = {
        "version": 1,
        "sequences": [
            {"ligand": {"id": "L1", "smiles": smiles}},
        ],
    }
    target = parse_boltz_schema("test", schema, ccd={}, boltz_2=True)
    return target.structure


class TestAtomNaming:
    """Per-element sequential naming must stay within the 4-char PDB limit.

    The naming scheme uses ELEMENT + sequential_number (e.g. C1, BR1),
    ensuring names stay within 4 characters even for large molecules.
    """

    @pytest.fixture(autouse=True)
    def _check_schema_deps(self):
        """Skip all tests if parse_boltz_schema deps are missing."""
        try:
            from boltz.data.parse.schema import parse_boltz_schema  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Cannot import parse_boltz_schema: {e}")

    def test_large_molecule_no_overflow(self):
        """No atom name exceeds 4 chars for large molecules."""
        smiles = "Br" + "C" * 50
        structure = _parse_ligand_smiles(smiles)

        for atom in structure.atoms:
            name = str(atom["name"])
            assert len(name) <= 4, f"Name '{name}' exceeds 4 chars"

    def test_names_are_unique(self):
        """All atom names must be unique within a molecule."""
        smiles = "Br" + "C" * 50
        structure = _parse_ligand_smiles(smiles)

        names = [str(a["name"]) for a in structure.atoms]
        assert len(names) == len(set(names)), (
            f"Duplicate atom names: {[n for n in names if names.count(n) > 1]}"
        )

    def test_names_are_per_element_sequential(self):
        """Atom names follow ELEMENT + sequential_number pattern."""
        smiles = "Br" + "C" * 50
        structure = _parse_ligand_smiles(smiles)

        for atom in structure.atoms:
            name = str(atom["name"])
            assert re.match(r"^[A-Z]{1,2}\d+$", name), (
                f"Name '{name}' doesn't match ELEMENT+NUMBER pattern"
            )

    def test_element_field_matches_name(self):
        """Stored element field is consistent with the atom name prefix."""
        smiles = "Br" + "C" * 50
        structure = _parse_ligand_smiles(smiles)
        periodic_table = Chem.GetPeriodicTable()

        for atom in structure.atoms:
            name = str(atom["name"])
            element_num = atom["element"].item()
            symbol = periodic_table.GetElementSymbol(element_num).upper()
            assert name.startswith(symbol), (
                f"Name '{name}' should start with '{symbol}' (Z={element_num})"
            )

    def test_small_molecule(self):
        """Small molecules (e.g. aspirin) get valid names."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        structure = _parse_ligand_smiles(smiles)

        for atom in structure.atoms:
            name = str(atom["name"])
            assert 2 <= len(name) <= 4, f"Name '{name}' has invalid length"

    def test_chlorine_in_large_molecule(self):
        """2-char elements don't overflow in large molecules."""
        smiles = "Cl" + "C" * 60 + "Cl"
        structure = _parse_ligand_smiles(smiles)

        for atom in structure.atoms:
            name = str(atom["name"])
            assert len(name) <= 4, f"Name '{name}' exceeds 4 chars"

    def test_old_scheme_would_overflow(self):
        """Demonstrate that a naive global-rank naming scheme overflows."""
        smiles = "Br" + "C" * 50
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        canonical_order = AllChem.CanonicalRankAtoms(mol)

        overflow_found = False
        for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
            old_name = atom.GetSymbol().upper() + str(can_idx + 1)
            if len(old_name) > 4:
                overflow_found = True
                break

        assert overflow_found, (
            "Expected naive naming scheme to overflow for this molecule"
        )


class TestChiralConstraints:
    """compute_chiral_atom_constraints must detect chiral centers."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        """Skip if schema deps are missing."""
        try:
            from boltz.data.parse.schema import compute_chiral_atom_constraints  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Cannot import schema: {e}")

    def test_finds_chiral_center_in_alanine(self):
        """L-Alanine has one chiral center — must produce at least one constraint."""
        from boltz.data.parse.schema import compute_chiral_atom_constraints

        mol = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        idx_map = {i: i for i in range(mol.GetNumAtoms())}

        constraints = compute_chiral_atom_constraints(mol, idx_map)
        assert len(constraints) > 0, (
            "Expected chiral constraints for L-Alanine, got none"
        )

    def test_no_constraints_for_achiral(self):
        """Glycine (no chiral center) produces no constraints."""
        from boltz.data.parse.schema import compute_chiral_atom_constraints

        mol = Chem.MolFromSmiles("NCC(=O)O")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        idx_map = {i: i for i in range(mol.GetNumAtoms())}

        constraints = compute_chiral_atom_constraints(mol, idx_map)
        assert len(constraints) == 0


class TestLeavingAtoms:
    """parse_ccd_residue must respect drop_leaving_atoms for multi-CCD ligands."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        """Skip if schema deps are missing."""
        try:
            from boltz.data.parse.schema import parse_ccd_residue  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Cannot import schema: {e}")

    @staticmethod
    def _make_mol_with_leaving_atoms():
        """Build a 3-heavy-atom mol with name/leaving_atom props (mimics CCD).

        Returns a molecule with 3 carbons: C1 (not leaving), C2 (not leaving),
        C3 (leaving atom=1).
        """
        mol = Chem.RWMol()
        for i in range(3):
            idx = mol.AddAtom(Chem.Atom(6))
            mol.GetAtomWithIdx(idx).SetProp("name", f"C{i+1}")
            mol.GetAtomWithIdx(idx).SetProp(
                "leaving_atom", "1" if i == 2 else "0"
            )
        mol.AddBond(0, 1, Chem.BondType.SINGLE)
        mol.AddBond(1, 2, Chem.BondType.SINGLE)
        mol = mol.GetMol()

        conf = Chem.Conformer(3)
        for i in range(3):
            conf.SetAtomPosition(i, (float(i), 0.0, 0.0))
        mol.AddConformer(conf, assignId=True)
        return mol

    def test_keeps_all_atoms_when_disabled(self):
        """drop_leaving_atoms=False preserves all atoms."""
        from boltz.data.parse.schema import parse_ccd_residue

        mol = self._make_mol_with_leaving_atoms()
        residue = parse_ccd_residue("TST", mol, 0, drop_leaving_atoms=False)

        assert residue is not None
        assert len(residue.atoms) == 3

    def test_drops_leaving_atoms_when_enabled(self):
        """drop_leaving_atoms=True removes atoms marked as leaving."""
        from boltz.data.parse.schema import parse_ccd_residue

        mol = self._make_mol_with_leaving_atoms()
        residue_all = parse_ccd_residue("TST", mol, 0, drop_leaving_atoms=False)
        residue_trimmed = parse_ccd_residue("TST", mol, 0, drop_leaving_atoms=True)

        assert residue_all is not None
        assert residue_trimmed is not None
        assert len(residue_trimmed.atoms) < len(residue_all.atoms), (
            f"Expected fewer atoms with drop_leaving_atoms=True: "
            f"got {len(residue_trimmed.atoms)} vs {len(residue_all.atoms)}"
        )

    def test_multi_ccd_schema_path(self):
        """parse_boltz_schema with multi-CCD ligand drops leaving atoms."""
        from unittest.mock import patch

        from boltz.data.parse.schema import parse_boltz_schema

        mol = self._make_mol_with_leaving_atoms()
        ccd = {"TS1": mol, "TS2": mol}

        schema = {
            "version": 1,
            "sequences": [
                {"ligand": {"id": "L1", "ccd": ["TS1", "TS2"]}},
            ],
        }

        # Spy on parse_ccd_residue to verify drop_leaving_atoms=True
        calls = []
        original = __import__(
            "boltz.data.parse.schema", fromlist=["parse_ccd_residue"]
        ).parse_ccd_residue

        def spy(*args, **kwargs):
            calls.append(kwargs if kwargs else {"drop_leaving_atoms": args[3] if len(args) > 3 else False})
            return original(*args, **kwargs)

        with patch("boltz.data.parse.schema.parse_ccd_residue", side_effect=spy):
            parse_boltz_schema("test", schema, ccd=ccd, boltz_2=True)

        # Both CCD codes should have been parsed with drop_leaving_atoms=True
        assert len(calls) == 2
        for call in calls:
            assert call.get("drop_leaving_atoms", False) is True
