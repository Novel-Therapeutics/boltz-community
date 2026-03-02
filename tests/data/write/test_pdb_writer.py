"""Tests for boltz.data.write.pdb — to_pdb."""

import numpy as np
import pytest
import torch

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


def _make_boltz2_ligand_structure(atom_name_str="CA1", element=20):
    """Create a minimal boltz2 StructureV2 with 1 NONPOLYMER chain, 1 atom."""
    coords_3d = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)

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


class TestToPdb:
    """Tests for to_pdb."""

    def test_atom_records(self, minimal_structure):
        """Output contains ATOM records."""
        pdb_str = to_pdb(minimal_structure)
        lines = pdb_str.split("\n")
        atom_lines = [l for l in lines if l.startswith("ATOM")]
        assert len(atom_lines) > 0

    def test_ter_and_end(self, minimal_structure_with_ligand):
        """Output contains TER between chains and END at the end."""
        pdb_str = to_pdb(minimal_structure_with_ligand)
        lines = pdb_str.split("\n")
        ter_lines = [l for l in lines if l.strip().startswith("TER")]
        end_lines = [l for l in lines if l.strip().startswith("END")]
        assert len(ter_lines) >= 1
        assert len(end_lines) >= 1

    def test_column_alignment(self, minimal_structure):
        """PDB lines are 80 chars wide."""
        pdb_str = to_pdb(minimal_structure)
        lines = pdb_str.split("\n")
        for line in lines:
            if line.strip():
                assert len(line) == 80

    def test_plddt_to_bfactor(self, minimal_structure):
        """pLDDT values appear in B-factor column (cols 61-66)."""
        # 3 residues → 3 plddt values
        plddts = torch.tensor([0.85, 0.70, 0.95])
        pdb_str = to_pdb(minimal_structure, plddts=plddts)

        lines = [l for l in pdb_str.split("\n") if l.startswith("ATOM")]
        # First residue atoms should have B-factor ≈ 85.0
        first_atom = lines[0]
        bfactor_str = first_atom[60:66].strip()
        bfactor = float(bfactor_str)
        assert bfactor == pytest.approx(85.0, abs=0.1)

    def test_hetatm_for_ligands(self, minimal_structure_with_ligand):
        """NONPOLYMER chains use HETATM records."""
        pdb_str = to_pdb(minimal_structure_with_ligand)
        lines = pdb_str.split("\n")
        hetatm_lines = [l for l in lines if l.startswith("HETATM")]
        assert len(hetatm_lines) > 0


class TestElementDetermination:
    """PDB writer must correctly determine element from the stored atomic number.

    Two-character elements (Ca, Fe, Br, Cl) must not be truncated to
    single-character elements (C, F, B, C).
    """

    @pytest.mark.parametrize(
        "atom_name,element_num,expected",
        [
            ("CA1", 20, "CA"),
            ("FE1", 26, "FE"),
            ("BR1", 35, "BR"),
            ("CL1", 17, "CL"),
            ("C1", 6, "C"),
            ("O1", 8, "O"),
        ],
    )
    def test_element_field(self, atom_name, element_num, expected):
        """Element column (77-78) must match the stored atomic number."""
        structure = _make_boltz2_ligand_structure(atom_name_str=atom_name, element=element_num)
        pdb_str = to_pdb(structure, boltz2=True)

        hetatm_lines = [l for l in pdb_str.split("\n") if l.startswith("HETATM")]
        assert len(hetatm_lines) == 1

        element_field = hetatm_lines[0][76:78].strip()
        assert element_field == expected

    @pytest.mark.parametrize(
        "atom_name,element_num",
        [("CA1", 20), ("FE1", 26), ("BR1", 35), ("C1", 6), ("CL1", 17)],
    )
    def test_pdb_line_length(self, atom_name, element_num):
        """Every HETATM line must be exactly 80 characters."""
        structure = _make_boltz2_ligand_structure(atom_name_str=atom_name, element=element_num)
        pdb_str = to_pdb(structure, boltz2=True)
        hetatm_lines = [l for l in pdb_str.split("\n") if l.startswith("HETATM")]
        for line in hetatm_lines:
            assert len(line) == 80


class TestLigandResidueName:
    """PDB writer must use actual residue name for HETATM records, not hardcoded 'LIG'."""

    def test_ccd_residue_name_preserved(self):
        """HETATM records show the residue's actual name."""
        structure = _make_boltz2_ligand_structure(atom_name_str="C1", element=6)
        structure.residues[0]["name"] = "ATP"

        pdb_str = to_pdb(structure, boltz2=True)
        hetatm_lines = [l for l in pdb_str.split("\n") if l.startswith("HETATM")]
        assert len(hetatm_lines) == 1

        res_name = hetatm_lines[0][17:20].strip()
        assert res_name == "ATP"

    def test_lig_still_works(self):
        """Residues actually named 'LIG' still show 'LIG'."""
        structure = _make_boltz2_ligand_structure(atom_name_str="C1", element=6)
        pdb_str = to_pdb(structure, boltz2=True)
        hetatm_lines = [l for l in pdb_str.split("\n") if l.startswith("HETATM")]

        res_name = hetatm_lines[0][17:20].strip()
        assert res_name == "LIG"
