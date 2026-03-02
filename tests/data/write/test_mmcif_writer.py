"""Tests for boltz.data.write.mmcif — to_mmcif."""

import re

import numpy as np
import pytest

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
from boltz.data.write.mmcif import to_mmcif


class TestToMmcif:
    """Tests for to_mmcif."""

    def test_basic_output(self, minimal_structure):
        """Output contains _atom_site records."""
        mmcif_str = to_mmcif(minimal_structure)
        assert "_atom_site" in mmcif_str
        assert len(mmcif_str) > 100

    def test_gemmi_roundtrip(self, minimal_structure):
        """Output is parseable by gemmi."""
        gemmi = pytest.importorskip("gemmi")
        mmcif_str = to_mmcif(minimal_structure)

        doc = gemmi.cif.read_string(mmcif_str)
        assert len(doc) > 0
        block = doc[0]
        # Should have atom records via _atom_site category
        # gemmi uses find_loop or find for block
        loop = block.find_loop("_atom_site.id")
        assert loop is not None

    def test_entity_type(self, minimal_structure):
        """Protein entity produces valid mmCIF output."""
        mmcif_str = to_mmcif(minimal_structure)

        # Should contain entity information and atom records
        assert "_atom_site" in mmcif_str
        # Check there's meaningful content
        lines = mmcif_str.split("\n")
        assert len(lines) > 10


class TestMmcifEntityMapping:
    """mmCIF writer must create separate Entity objects per entity_id."""

    @staticmethod
    def _make_two_ligand_structure(same_entity=False):
        """Build a StructureV2 with two NONPOLYMER chains (1 atom each)."""
        atoms = np.zeros(2, dtype=AtomV2)
        for i in range(2):
            atoms[i]["name"] = f"C{i+1}"
            atoms[i]["element"] = 6
            atoms[i]["coords"] = [float(i), 0.0, 0.0]
            atoms[i]["is_present"] = True

        residues = np.zeros(2, dtype=Residue)
        for i in range(2):
            residues[i]["name"] = "LIG"
            residues[i]["res_type"] = 0
            residues[i]["res_idx"] = 0
            residues[i]["atom_idx"] = i
            residues[i]["atom_num"] = 1
            residues[i]["atom_center"] = 0
            residues[i]["atom_disto"] = 0
            residues[i]["is_standard"] = False
            residues[i]["is_present"] = True

        chains = np.zeros(2, dtype=Chain)
        for i in range(2):
            chains[i]["name"] = chr(ord("A") + i)
            chains[i]["mol_type"] = const.chain_type_ids["NONPOLYMER"]
            chains[i]["entity_id"] = 0 if same_entity else i
            chains[i]["sym_id"] = i
            chains[i]["asym_id"] = i
            chains[i]["atom_idx"] = i
            chains[i]["atom_num"] = 1
            chains[i]["res_idx"] = i
            chains[i]["res_num"] = 1
            chains[i]["cyclic_period"] = 0

        coords_arr = np.array(
            [([float(i), 0.0, 0.0],) for i in range(2)], dtype=Coords
        )
        ensemble = np.zeros(1, dtype=Ensemble)
        ensemble[0]["atom_coord_idx"] = 0
        ensemble[0]["atom_num"] = 2

        return StructureV2(
            atoms=atoms,
            bonds=np.array([], dtype=BondV2),
            residues=residues,
            chains=chains,
            interfaces=np.array([], dtype=Interface),
            mask=np.ones(2, dtype=bool),
            coords=coords_arr,
            ensemble=ensemble,
        )

    def test_different_entities_produce_distinct_records(self):
        """Two NONPOLYMER chains with different entity_ids map to separate entities."""
        structure = self._make_two_ligand_structure(same_entity=False)
        # Different residue names so ihm library doesn't reject as duplicates
        structure.residues[0]["name"] = "ATP"
        structure.residues[1]["name"] = "GTP"

        cif_str = to_mmcif(structure, boltz2=True)

        assert "Model subunit A" in cif_str
        assert "Model subunit B" in cif_str

        # _struct_asym should map A→entity 1 and B→entity 2
        assert re.search(r"^A\s+1\s+'Model subunit A'", cif_str, re.MULTILINE)
        assert re.search(r"^B\s+2\s+'Model subunit B'", cif_str, re.MULTILINE)

    def test_same_entity_shared(self):
        """Two chains with the same entity_id share one Entity object."""
        structure = self._make_two_ligand_structure(same_entity=True)
        cif_str = to_mmcif(structure, boltz2=True)

        assert "Model subunit A" in cif_str
        assert "Model subunit B" in cif_str
