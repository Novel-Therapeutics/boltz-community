"""Tests for boltz.data.write.mmcif — to_mmcif."""

import re
from dataclasses import replace

import numpy as np
import pytest

from boltz.data import const
from boltz.data.types import (
    AtomV2,
    BondV2,
    Chain,
    Connection,
    Coords,
    Ensemble,
    Interface,
    Residue,
    StructureV2,
)
from boltz.data.write.mmcif import _add_struct_conn_records, to_mmcif


def _struct_conn_rows(cif_str: str) -> list[dict[str, str]]:
    """Return struct_conn rows as dictionaries keyed by column name."""
    gemmi = pytest.importorskip("gemmi")
    block = gemmi.cif.read_string(cif_str)[0]
    item = block.find_loop_item("_struct_conn.id")
    assert item is not None
    loop = item.loop
    tags = [tag.replace("_struct_conn.", "") for tag in loop.tags]

    def value_at(row_idx: int, col_idx: int) -> str:
        value = loop[row_idx, col_idx]
        if value in {"?", "."}:
            return value
        return gemmi.cif.as_string(value)

    return [
        {tag: value_at(row_idx, col_idx) for col_idx, tag in enumerate(tags)}
        for row_idx in range(loop.length())
    ]


def _example_struct_conn_row(row_id: str = "covale1") -> tuple:
    """Return a minimal complete struct_conn row for helper tests."""
    return (
        row_id,
        "covale",
        "A",
        "ATP",
        1,
        "C1",
        "A",
        "ATP",
        1,
        "C1",
        "B",
        "GTP",
        1,
        "C2",
        "B",
        "GTP",
        1,
        "C2",
        "1.000",
    )


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

    def test_cross_residue_bonds_are_written_as_struct_conn(self):
        """Cross-residue covalent bonds should survive in mmCIF output."""
        gemmi = pytest.importorskip("gemmi")
        structure = self._make_two_ligand_structure(same_entity=False)
        structure.residues[0]["name"] = "ATP"
        structure.residues[1]["name"] = "GTP"
        structure = replace(
            structure,
            bonds=np.array(
                [(0, 1, 0, 1, 0, 1, const.bond_type_ids["COVALENT"])],
                dtype=BondV2,
            ),
        )

        cif_str = to_mmcif(structure, boltz2=True)

        assert "_struct_conn.id" in cif_str
        assert "_struct_conn.ptnr1_auth_atom_id" in cif_str
        assert "_struct_conn.ptnr2_auth_atom_id" in cif_str
        rows = _struct_conn_rows(cif_str)
        assert len(rows) == 1
        row = rows[0]
        assert row["ptnr1_label_atom_id"] == "C1"
        assert row["ptnr1_auth_atom_id"] == "C1"
        assert row["ptnr2_label_atom_id"] == "C2"
        assert row["ptnr2_auth_atom_id"] == "C2"
        assert float(row["pdbx_dist_value"]) == pytest.approx(1.0)

        parsed = gemmi.read_structure_string(cif_str, format=gemmi.CoorFormat.Mmcif)
        assert len(parsed.connections) == 1
        connection = parsed.connections[0]
        assert connection.partner1.chain_name == "A"
        assert connection.partner1.atom_name == "C1"
        assert connection.partner2.chain_name == "B"
        assert connection.partner2.atom_name == "C2"

    def test_boltz1_connections_are_written_as_struct_conn(self, minimal_structure):
        """Boltz-1 Structure.connections should survive in mmCIF output."""
        gemmi = pytest.importorskip("gemmi")
        structure = replace(
            minimal_structure,
            connections=np.array([(0, 0, 0, 1, 2, 5)], dtype=Connection),
        )

        cif_str = to_mmcif(structure, boltz2=False)

        assert "_struct_conn.id" in cif_str
        rows = _struct_conn_rows(cif_str)
        assert len(rows) == 1
        row = rows[0]
        expected_distance = np.linalg.norm(
            structure.atoms[2]["coords"] - structure.atoms[5]["coords"]
        )
        assert row["ptnr1_label_atom_id"] == "C"
        assert row["ptnr1_auth_atom_id"] == "C"
        assert row["ptnr2_label_atom_id"] == "N"
        assert row["ptnr2_auth_atom_id"] == "N"
        assert float(row["pdbx_dist_value"]) == pytest.approx(expected_distance)

        parsed = gemmi.read_structure_string(cif_str, format=gemmi.CoorFormat.Mmcif)
        assert len(parsed.connections) == 1
        connection = parsed.connections[0]
        assert connection.partner1.chain_name == "A"
        assert connection.partner1.res_id.seqid.num == 1
        assert connection.partner1.atom_name == "C"
        assert connection.partner2.chain_name == "A"
        assert connection.partner2.res_id.seqid.num == 2
        assert connection.partner2.atom_name == "N"

    def test_existing_struct_conn_loop_is_preserved(self):
        """Adding rows should append to an existing struct_conn loop."""
        base = """\
data_model
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.pdbx_dist_value
'covale1' covale OLD 9.999
#
"""
        new_row = _example_struct_conn_row()

        cif_str = _add_struct_conn_records(base, [new_row, new_row])

        rows = _struct_conn_rows(cif_str)
        assert len(rows) == 3
        assert rows[0]["id"] == "covale1"
        assert rows[0]["ptnr1_label_atom_id"] == "OLD"
        assert rows[0]["pdbx_dist_value"] == "9.999"
        assert rows[0]["ptnr1_auth_atom_id"] == "?"
        assert rows[1]["id"] == "covale2"
        assert rows[1]["ptnr1_label_atom_id"] == "C1"
        assert rows[1]["ptnr1_auth_atom_id"] == "C1"
        assert rows[1]["ptnr2_auth_atom_id"] == "C2"
        assert rows[1]["pdbx_dist_value"] == "1.000"
        assert rows[2]["id"] == "covale3"
        assert rows[2]["ptnr1_label_atom_id"] == "C1"

    def test_struct_conn_records_require_single_data_block(self):
        """Struct_conn append should fail loudly for multi-block mmCIF."""
        base = """\
data_model_1
#
data_model_2
#
"""

        with pytest.raises(ValueError, match="exactly one mmCIF data block"):
            _add_struct_conn_records(base, [_example_struct_conn_row()])

    def test_struct_conn_records_reject_unterminated_text_field(self):
        """Malformed mmCIF text fields should fail before appending records."""
        base = """\
data_model
_audit.details
;
_struct_conn.id appears here as prose, not as a tag.
#
"""

        with pytest.raises(ValueError, match="unterminated text field"):
            _add_struct_conn_records(base, [_example_struct_conn_row()])

    def test_struct_conn_text_field_does_not_disable_fast_path(self, monkeypatch):
        """Mentioning struct_conn inside a text field should not trigger parsing."""
        base = """\
data_model
_audit.details
;
_struct_conn.id appears here as prose, not as a tag.
;
#
"""

        def fail_read_string(_mmcif: str) -> None:
            msg = "fast path should not parse this mmCIF"
            raise AssertionError(msg)

        with monkeypatch.context() as m:
            m.setattr(
                "boltz.data.write.mmcif.cif.read_string",
                fail_read_string,
            )
            cif_str = _add_struct_conn_records(base, [_example_struct_conn_row()])

        rows = _struct_conn_rows(cif_str)
        assert len(rows) == 1
        assert rows[0]["id"] == "covale1"
