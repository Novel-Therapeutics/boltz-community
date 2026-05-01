"""Tests for boltz.data.parse.schema — atom naming, chirality, and leaving atoms."""

import logging
import re
import textwrap
from pathlib import Path

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
    target = parse_boltz_schema(Path("test.yaml"), schema, ccd={}, boltz_2=True)
    return target.structure


def _mock_residue_mol(res_name, _mols, _moldir):
    """Build a small reference residue mol with atom-name properties."""
    from boltz.data import const

    mol = Chem.RWMol()
    coords = []
    for atom_name in const.ref_atoms[res_name]:
        if atom_name.startswith("N"):
            atomic_num = 7
        elif atom_name.startswith("O"):
            atomic_num = 8
        elif atom_name.startswith("S"):
            atomic_num = 16
        elif atom_name.startswith("P"):
            atomic_num = 15
        else:
            atomic_num = 6
        idx = mol.AddAtom(Chem.Atom(atomic_num))
        mol.GetAtomWithIdx(idx).SetProp("name", atom_name)
        coords.append((float(idx), 0.0, 0.0))
        if idx > 0:
            mol.AddBond(idx - 1, idx, Chem.BondType.SINGLE)

    mol = mol.GetMol()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for idx, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(idx, (x, y, z))
    mol.AddConformer(conformer)
    return mol


def _make_named_mol(atom_specs, bonds):
    """Build a small CCD-like molecule with named heavy atoms."""
    mol = Chem.RWMol()
    for atom_name, atomic_num in atom_specs:
        idx = mol.AddAtom(Chem.Atom(atomic_num))
        atom = mol.GetAtomWithIdx(idx)
        atom.SetProp("name", atom_name)
        atom.SetProp("leaving_atom", "0")
    for idx_1, idx_2, bond_type in bonds:
        mol.AddBond(idx_1, idx_2, bond_type)

    mol = mol.GetMol()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for idx in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(idx, (float(idx), 0.0, 0.0))
    mol.AddConformer(conformer)
    return mol


def _build_atom_id_map(structure):
    """Map (residue index, atom name) to atom table index."""
    return {
        (int(res["res_idx"]), str(atom["name"])): atom_idx
        for res in structure.residues
        for atom_idx, atom in enumerate(
            structure.atoms[
                int(res["atom_idx"]) : int(res["atom_idx"] + res["atom_num"])
            ],
            start=int(res["atom_idx"]),
        )
    }


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
            from boltz.data.parse.schema import (
                compute_chiral_atom_constraints,  # noqa: F401
            )
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
            parse_boltz_schema(Path("test.yaml"), schema, ccd=ccd, boltz_2=True)

        # Both CCD codes should have been parsed with drop_leaving_atoms=True
        assert len(calls) == 2
        for call in calls:
            assert call.get("drop_leaving_atoms", False) is True


class TestExplicitBondConstraints:
    """User-provided bonds should survive as graph and geometry constraints."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        """Skip if parser deps are missing."""
        try:
            from boltz.data.parse.schema import parse_boltz_schema  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Cannot import parse_boltz_schema: {e}")

    def test_modified_residue_bond_adds_geometry_bound(self, tmp_path):
        """Cross-residue bond constraints should get a bond-length bound."""
        from boltz.data import const
        from boltz.data.parse.schema import (
            _estimate_covalent_bond_length_with_fallback,
            parse_boltz_schema,
        )

        ace = _make_named_mol(
            [("C", 6), ("O", 8), ("CH3", 6)],
            [(0, 1, Chem.BondType.DOUBLE), (0, 2, Chem.BondType.SINGLE)],
        )
        cy3 = _make_named_mol(
            [("N", 7), ("CA", 6), ("C", 6), ("O", 8), ("CB", 6), ("SG", 16)],
            [
                (0, 1, Chem.BondType.SINGLE),
                (1, 2, Chem.BondType.SINGLE),
                (1, 4, Chem.BondType.SINGLE),
                (2, 3, Chem.BondType.DOUBLE),
                (4, 5, Chem.BondType.SINGLE),
            ],
        )
        schema = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "B",
                        "sequence": "GC",
                        "modifications": [
                            {"position": 1, "ccd": "ACE"},
                            {"position": 2, "ccd": "CY3"},
                        ],
                    }
                },
            ],
            "constraints": [
                {
                    "bond": {
                        "atom1": ["B", 1, "CH3"],
                        "atom2": ["B", 2, "SG"],
                    }
                }
            ],
        }

        target = parse_boltz_schema(
            Path("issue675.yaml"),
            schema,
            ccd={"ACE": ace, "CY3": cy3},
            mol_dir=tmp_path,
            boltz_2=True,
        )
        structure = target.structure
        atom_ids = _build_atom_id_map(structure)
        ch3 = atom_ids[(0, "CH3")]
        sg = atom_ids[(1, "SG")]

        matching_bonds = [
            bond
            for bond in structure.bonds
            if {int(bond["atom_1"]), int(bond["atom_2"])} == {ch3, sg}
        ]
        assert len(matching_bonds) == 1
        assert int(matching_bonds[0]["type"]) == const.bond_type_ids["COVALENT"]

        bounds = target.residue_constraints.rdkit_bounds_constraints
        matching_bounds = [
            bound
            for bound in bounds
            if set(map(int, bound["atom_idxs"])) == {ch3, sg}
        ]
        assert len(matching_bounds) == 1
        assert bool(matching_bounds[0]["is_bond"])
        assert not bool(matching_bounds[0]["is_angle"])
        expected, used_fallback = _estimate_covalent_bond_length_with_fallback(
            6, 16
        )
        assert not used_fallback
        assert matching_bounds[0]["lower_bound"] == pytest.approx(
            expected - 0.05,
            abs=1e-3,
        )
        assert matching_bounds[0]["upper_bound"] == pytest.approx(
            expected + 0.05,
            abs=1e-3,
        )

    def test_unknown_bond_atom_raises_valueerror(self, tmp_path):
        """Bond constraints should report typos as ValueError."""
        from boltz.data.parse.schema import parse_boltz_schema

        ace = _make_named_mol(
            [("C", 6), ("O", 8), ("CH3", 6)],
            [(0, 1, Chem.BondType.DOUBLE), (0, 2, Chem.BondType.SINGLE)],
        )
        schema = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "B",
                        "sequence": "G",
                        "modifications": [{"position": 1, "ccd": "ACE"}],
                    }
                },
            ],
            "constraints": [
                {"bond": {"atom1": ["B", 1, "CH3"], "atom2": ["B", 1, "NOPE"]}}
            ],
        }

        with pytest.raises(ValueError, match="unknown atom"):
            parse_boltz_schema(
                Path("bad_bond.yaml"),
                schema,
                ccd={"ACE": ace},
                mol_dir=tmp_path,
                boltz_2=True,
            )

    def test_unknown_element_bond_uses_fallback_bound(self, tmp_path, caplog):
        """Unknown covalent radii should fall back to a generic bond length."""
        from boltz.data.parse.schema import parse_boltz_schema

        caplog.set_level(logging.WARNING, logger="boltz.data.parse.schema")
        dummy = _make_named_mol([("DU", 0)], [])
        carbon = _make_named_mol([("C1", 6)], [])
        schema = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "B",
                        "sequence": "GC",
                        "modifications": [
                            {"position": 1, "ccd": "DUM"},
                            {"position": 2, "ccd": "CAR"},
                        ],
                    }
                },
            ],
            "constraints": [
                {"bond": {"atom1": ["B", 1, "DU"], "atom2": ["B", 2, "C1"]}}
            ],
        }

        target = parse_boltz_schema(
            Path("unknown_element.yaml"),
            schema,
            ccd={"DUM": dummy, "CAR": carbon},
            mol_dir=tmp_path,
            boltz_2=True,
        )

        structure = target.structure
        atom_ids = _build_atom_id_map(structure)
        du = atom_ids[(0, "DU")]
        c1 = atom_ids[(1, "C1")]
        bounds = target.residue_constraints.rdkit_bounds_constraints
        fallback_bounds = [
            bound
            for bound in bounds
            if set(map(int, bound["atom_idxs"])) == {du, c1}
        ]
        assert len(fallback_bounds) == 1
        assert fallback_bounds[0]["lower_bound"] == pytest.approx(1.5, abs=1e-3)
        assert fallback_bounds[0]["upper_bound"] == pytest.approx(2.5, abs=1e-3)
        assert any(
            "Using fallback covalent bond length for explicit bond DU-C1"
            in record.message
            for record in caplog.records
        )


class TestTemplatePaths:
    """Template paths should be resolved relative to the YAML file."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        """Skip if parser deps are missing."""
        try:
            from boltz.data.parse.yaml import parse_yaml  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Cannot import parse_yaml: {e}")

    @staticmethod
    def _write_template_pdb(path: Path) -> None:
        """Write a minimal atom-only PDB that parses as a protein template."""
        path.write_text(
            """\
ATOM      1  N   ALA A   1      11.104  13.207  14.101  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207  14.101  1.00 20.00           C
ATOM      3  C   ALA A   1      13.000  14.500  14.800  1.00 20.00           C
ATOM      4  O   ALA A   1      12.300  15.500  14.700  1.00 20.00           O
ATOM      5  CB  ALA A   1      13.100  12.000  14.900  1.00 20.00           C
TER
END
"""
        )

    def test_relative_template_pdb_path_uses_yaml_parent(self, tmp_path, monkeypatch):
        """Template PDB paths should not depend on the current working directory."""
        from boltz.data.parse.yaml import parse_yaml

        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        template_path = input_dir / "template.pdb"
        yaml_path = input_dir / "input.yaml"
        self._write_template_pdb(template_path)
        yaml_path.write_text(
            textwrap.dedent(
                """\
                version: 1
                sequences:
                  - protein:
                      id: A
                      sequence: A
                      msa: empty
                templates:
                  - pdb: ./template.pdb
                    chain_id: A
                """
            )
        )

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("boltz.data.parse.schema.get_mol", _mock_residue_mol)
        monkeypatch.setattr("boltz.data.parse.mmcif.get_mol", _mock_residue_mol)

        target = parse_yaml(yaml_path, ccd={}, mol_dir=tmp_path, boltz2=True)

        assert len(target.record.templates or []) == 1
        assert list(target.templates or {}) == ["template"]


class TestAffinityParsing:
    """Repeated ligand copies should remain valid affinity inputs."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        """Skip if parser deps are missing."""
        try:
            from boltz.data.parse.schema import parse_boltz_schema  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Cannot import schema: {e}")

    def test_repeated_binder_keeps_requested_chain(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        """A repeated ligand binder should preserve the requested chain name."""
        from boltz.data.parse.schema import parse_boltz_schema

        ligand = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(ligand, randomSeed=42)
        for idx, atom in enumerate(ligand.GetAtoms()):
            atom.SetProp("name", f"C{idx + 1}")
            atom.SetProp("leaving_atom", "0")

        ccd = {"TST": ligand}

        def _mock_get_mol(res_name, mols, moldir):
            if res_name in mols:
                return mols[res_name]
            return _mock_residue_mol(res_name, mols, moldir)

        monkeypatch.setattr("boltz.data.parse.schema.get_mol", _mock_get_mol)

        schema = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": "AA", "msa": "empty"}},
                {"ligand": {"id": ["L1", "L2"], "ccd": "TST"}},
            ],
            "properties": [
                {"affinity": {"binder": "L1"}},
            ],
        }

        target = parse_boltz_schema(
            Path("test.yaml"),
            schema,
            ccd=ccd,
            mol_dir=tmp_path,
            boltz_2=True,
        )

        assert target.record.affinity is not None
        assert target.record.affinity.chain_name == "L1"
        ligand_chains = [
            chain for chain in target.record.chains if chain.entity_id == 1
        ]
        assert [chain.chain_name for chain in ligand_chains] == ["L1", "L2"]
