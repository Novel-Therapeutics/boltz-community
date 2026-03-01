"""Tests for specific bug fixes cherry-picked or developed in boltz-community.

Each test class targets a specific upstream issue:
- TestSubsampleMsaDefault: upstream #628
- TestElementDetermination: upstream #458
- TestAtomNamingOverflow: upstream #494
"""

import re

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from boltz.data import const
from boltz.data.types import (
    AtomV2,
    BondV2,
    Chain,
    Coords,
    Ensemble,
    Interface,
    Residue,
    StructureV2,
)
from boltz.data.write.pdb import to_pdb


def _parse_ligand_smiles(smiles: str) -> StructureV2:
    """Run a SMILES string through the full production parse path.

    Calls ``parse_boltz_schema`` with ``boltz_2=True`` and returns
    the resulting ``StructureV2``.  Import is deferred to avoid
    collection failures when Bio/chembl_structure_pipeline are absent.
    """
    from boltz.data.parse.schema import parse_boltz_schema

    schema = {
        "version": 1,
        "sequences": [
            {"ligand": {"id": "L1", "smiles": smiles}},
        ],
    }
    target = parse_boltz_schema("test", schema, ccd={}, boltz_2=True)
    return target.structure


# ---------------------------------------------------------------------------
# Bug #628: --subsample_msa flag defaulted to False instead of True
# ---------------------------------------------------------------------------


class TestSubsampleMsaDefault:
    """The --subsample_msa CLI option must default to True (upstream #628).

    The old ``is_flag=True`` Click declaration made the default False when
    the flag was absent, contradicting the documented "Default is True".
    """

    @pytest.fixture(autouse=True)
    def _import_predict(self):
        """Import predict command, skipping if heavy deps are missing."""
        try:
            from boltz.main import predict
            self.predict = predict
        except ImportError as e:
            pytest.skip(f"Cannot import boltz.main: {e}")

    def test_default_is_true(self):
        """predict command's subsample_msa param defaults to True."""
        for param in self.predict.params:
            if param.name == "subsample_msa":
                assert param.default is True, (
                    f"subsample_msa default should be True, got {param.default}"
                )
                return
        pytest.fail("subsample_msa parameter not found on predict command")

    def test_is_boolean_flag_pair(self):
        """subsample_msa should support --subsample_msa / --no_subsample_msa."""
        for param in self.predict.params:
            if param.name == "subsample_msa":
                # Boolean flag pairs in Click have is_flag=True and secondary=True
                assert param.is_flag, "subsample_msa should be a flag"
                assert param.secondary, (
                    "subsample_msa should be a boolean flag pair "
                    "(--subsample_msa/--no_subsample_msa)"
                )
                return
        pytest.fail("subsample_msa parameter not found on predict command")


# ---------------------------------------------------------------------------
# Bug #458: Calcium (Ca) misidentified as Carbon (C) in PDB writer
# ---------------------------------------------------------------------------


def _make_boltz2_ligand_structure(atom_name_str="CA1", element=20):
    """Create a minimal boltz2 StructureV2 with 1 NONPOLYMER chain, 1 atom.

    Parameters
    ----------
    atom_name_str : str
        The 1-4 character atom name (e.g. "CA1", "FE1", "C1").
    element : int
        The atomic number (e.g. 20 for Ca, 26 for Fe, 6 for C).

    """
    coords_3d = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)

    # AtomV2 uses Unicode string names (not shifted-ASCII int arrays)
    atoms = np.zeros(1, dtype=AtomV2)
    atoms[0]["name"] = atom_name_str
    atoms[0]["element"] = element
    atoms[0]["coords"] = coords_3d[0]
    atoms[0]["is_present"] = True
    atoms[0]["bfactor"] = 0.0
    atoms[0]["plddt"] = 0.0

    residues = np.zeros(1, dtype=Residue)
    residues[0]["name"] = "LIG"
    residues[0]["res_type"] = 0
    residues[0]["res_idx"] = 0
    residues[0]["atom_idx"] = 0
    residues[0]["atom_num"] = 1
    residues[0]["atom_center"] = 0
    residues[0]["atom_disto"] = 0
    residues[0]["is_standard"] = False
    residues[0]["is_present"] = True

    chains = np.zeros(1, dtype=Chain)
    chains[0]["name"] = "A"
    chains[0]["mol_type"] = const.chain_type_ids["NONPOLYMER"]
    chains[0]["entity_id"] = 0
    chains[0]["sym_id"] = 0
    chains[0]["asym_id"] = 0
    chains[0]["atom_idx"] = 0
    chains[0]["atom_num"] = 1
    chains[0]["res_idx"] = 0
    chains[0]["res_num"] = 1
    chains[0]["cyclic_period"] = 0

    # StructureV2 also requires coords and ensemble arrays
    coords_arr = np.array([(coords_3d[0],)], dtype=Coords)
    ensemble = np.zeros(1, dtype=Ensemble)
    ensemble[0]["atom_coord_idx"] = 0
    ensemble[0]["atom_num"] = 1

    return StructureV2(
        atoms=atoms,
        bonds=np.array([], dtype=BondV2),
        residues=residues,
        chains=chains,
        interfaces=np.array([], dtype=Interface),
        mask=np.ones(1, dtype=bool),
        coords=coords_arr,
        ensemble=ensemble,
    )


class TestElementDetermination:
    """PDB writer must correctly determine element for ligand atoms (upstream #458).

    The old boltz2 code path inferred element from the atom name string,
    which failed for 2-char elements like Ca, Fe, Br (e.g. "CA1" was
    misidentified as Carbon via ``ambiguous_atoms["CA"]["*"] == "C"``).

    The fix stores the atomic number in the AtomV2 ``element`` field,
    so the writer uses the stored value directly via the periodic table.
    """

    def test_calcium_element_boltz2(self):
        """Calcium atom 'CA1' must have element 'CA' in PDB output, not 'C'."""
        structure = _make_boltz2_ligand_structure(atom_name_str="CA1", element=20)
        pdb_str = to_pdb(structure, boltz2=True)

        hetatm_lines = [l for l in pdb_str.split("\n") if l.startswith("HETATM")]
        assert len(hetatm_lines) == 1

        # PDB element is right-justified in columns 77-78 (1-indexed)
        # With 80-char lines: 0-indexed [76:78]
        element_field = hetatm_lines[0][76:78].strip()
        assert element_field == "CA", (
            f"Calcium should have element 'CA', got '{element_field}'"
        )

    def test_iron_element_boltz2(self):
        """Iron atom 'FE1' must have element 'FE', not 'F'."""
        structure = _make_boltz2_ligand_structure(atom_name_str="FE1", element=26)
        pdb_str = to_pdb(structure, boltz2=True)

        hetatm_lines = [l for l in pdb_str.split("\n") if l.startswith("HETATM")]
        element_field = hetatm_lines[0][76:78].strip()
        assert element_field == "FE", (
            f"Iron should have element 'FE', got '{element_field}'"
        )

    def test_bromine_element_boltz2(self):
        """Bromine atom 'BR1' must have element 'BR', not 'B'."""
        structure = _make_boltz2_ligand_structure(atom_name_str="BR1", element=35)
        pdb_str = to_pdb(structure, boltz2=True)

        hetatm_lines = [l for l in pdb_str.split("\n") if l.startswith("HETATM")]
        element_field = hetatm_lines[0][76:78].strip()
        assert element_field == "BR", (
            f"Bromine should have element 'BR', got '{element_field}'"
        )

    def test_carbon_element_boltz2(self):
        """Carbon atom 'C1' should still work correctly."""
        structure = _make_boltz2_ligand_structure(atom_name_str="C1", element=6)
        pdb_str = to_pdb(structure, boltz2=True)

        hetatm_lines = [l for l in pdb_str.split("\n") if l.startswith("HETATM")]
        element_field = hetatm_lines[0][76:78].strip()
        assert element_field == "C", (
            f"Carbon should have element 'C', got '{element_field}'"
        )

    def test_oxygen_element_boltz2(self):
        """Oxygen atom 'O1' must have element 'O'."""
        structure = _make_boltz2_ligand_structure(atom_name_str="O1", element=8)
        pdb_str = to_pdb(structure, boltz2=True)

        hetatm_lines = [l for l in pdb_str.split("\n") if l.startswith("HETATM")]
        element_field = hetatm_lines[0][76:78].strip()
        assert element_field == "O", (
            f"Oxygen should have element 'O', got '{element_field}'"
        )

    def test_chlorine_element_boltz2(self):
        """Chlorine atom 'CL1' must have element 'CL', not 'C'."""
        structure = _make_boltz2_ligand_structure(atom_name_str="CL1", element=17)
        pdb_str = to_pdb(structure, boltz2=True)

        hetatm_lines = [l for l in pdb_str.split("\n") if l.startswith("HETATM")]
        element_field = hetatm_lines[0][76:78].strip()
        assert element_field == "CL", (
            f"Chlorine should have element 'CL', got '{element_field}'"
        )

    def test_pdb_line_length(self):
        """Every HETATM line must be exactly 80 characters."""
        for name, elem in [("CA1", 20), ("FE1", 26), ("BR1", 35), ("C1", 6), ("CL1", 17)]:
            structure = _make_boltz2_ligand_structure(atom_name_str=name, element=elem)
            pdb_str = to_pdb(structure, boltz2=True)
            hetatm_lines = [l for l in pdb_str.split("\n") if l.startswith("HETATM")]
            for line in hetatm_lines:
                assert len(line) == 80, (
                    f"PDB line for {name} must be 80 chars, got {len(line)}"
                )


# ---------------------------------------------------------------------------
# Bug #494: atom names > 4 chars crash ligand processing
# ---------------------------------------------------------------------------


class TestAtomNamingOverflow:
    """Per-element sequential naming avoids 4-char PDB overflow (upstream #494).

    The old scheme used ``element + global_canonical_rank``, which overflows
    for molecules with >99 total atoms when 2-char elements are present
    (e.g. Br with rank 100 -> "BR100", 5 chars).  The fix uses per-element
    sequential numbering, which stays within 4 chars up to 999 atoms for
    1-char elements and 99 atoms for 2-char elements.

    Tests exercise the production path through ``parse_boltz_schema``
    to catch regressions even if the implementation changes.
    """

    @pytest.fixture(autouse=True)
    def _check_schema_deps(self):
        """Skip all tests if parse_boltz_schema deps are missing."""
        try:
            from boltz.data.parse.schema import parse_boltz_schema  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Cannot import parse_boltz_schema: {e}")

    def test_old_scheme_would_overflow(self):
        """Demonstrate that the old naming scheme fails on a large molecule."""
        # 1-bromopentacontane: Br + 50 C + 101 H = 152 atoms
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
            "Expected old naming scheme to overflow for this molecule"
        )

    def test_large_molecule_no_overflow(self):
        """parse_boltz_schema produces no atom name > 4 chars for large molecules."""
        smiles = "Br" + "C" * 50
        structure = _parse_ligand_smiles(smiles)

        for atom in structure.atoms:
            name = str(atom["name"])
            assert len(name) <= 4, f"Name '{name}' exceeds 4 chars"

    def test_names_are_unique(self):
        """All atom names produced by parse_boltz_schema must be unique."""
        smiles = "Br" + "C" * 50
        structure = _parse_ligand_smiles(smiles)

        names = [str(a["name"]) for a in structure.atoms]
        assert len(names) == len(set(names)), (
            f"Duplicate atom names: {[n for n in names if names.count(n) > 1]}"
        )

    def test_names_are_per_element_sequential(self):
        """Atom names must follow ELEMENT + sequential_number pattern."""
        smiles = "Br" + "C" * 50
        structure = _parse_ligand_smiles(smiles)

        for atom in structure.atoms:
            name = str(atom["name"])
            assert re.match(r"^[A-Z]{1,2}\d+$", name), (
                f"Name '{name}' doesn't match ELEMENT+NUMBER pattern"
            )

    def test_element_field_matches_name(self):
        """The stored element field must be consistent with the atom name prefix."""
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

    def test_small_molecule_still_works(self):
        """Small molecules (e.g. aspirin) should also get valid names."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        structure = _parse_ligand_smiles(smiles)

        for atom in structure.atoms:
            name = str(atom["name"])
            assert 2 <= len(name) <= 4, f"Name '{name}' has invalid length"

    def test_chlorine_in_large_molecule(self):
        """Chlorine (2-char element) should not overflow in large molecules."""
        smiles = "Cl" + "C" * 60 + "Cl"
        structure = _parse_ligand_smiles(smiles)

        for atom in structure.atoms:
            name = str(atom["name"])
            assert len(name) <= 4, f"Name '{name}' exceeds 4 chars"
