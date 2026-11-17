"""Tests for boltz.data.write.pdb — to_pdb."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from boltz.data import const
from boltz.data.types import (
    Atom,
    AtomV2,
    Bond,
    BondV2,
    Chain,
    Connection,
    Coords,
    Ensemble,
    Interface,
    Residue,
    Structure,
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


def _conect_lines(pdb_str):
    return [l.rstrip() for l in pdb_str.split("\n") if l.startswith("CONECT")]


def _connectivity_structure(boltz2, atom_counts=(4,), bonds=()):
    """Build one residue per chain in either atom format, with source-index bonds."""
    atoms = np.zeros(sum(atom_counts), dtype=AtomV2 if boltz2 else Atom)
    for i, atom in enumerate(atoms):
        name = f"C{i + 1}"
        atom["name"] = name if boltz2 else [ord(c) - 32 for c in name.ljust(4)]
        atom["element"] = 6
        atom["coords"] = (float(i), 0, 0)
        atom["is_present"] = True
    residues = np.zeros(len(atom_counts), dtype=Residue)
    chains = np.zeros(len(atom_counts), dtype=Chain)
    start = 0
    for i, count in enumerate(atom_counts):
        residues[i]["name"] = "ALA"
        residues[i]["res_idx"] = i
        residues[i]["atom_idx"] = start
        residues[i]["atom_num"] = count
        residues[i]["is_present"] = True
        chains[i]["name"] = chr(ord("A") + i)
        chains[i]["mol_type"] = const.chain_type_ids["PROTEIN"]
        chains[i]["asym_id"] = i
        chains[i]["res_idx"] = i
        chains[i]["res_num"] = 1
        chains[i]["atom_idx"] = start
        chains[i]["atom_num"] = count
        start += count
    bond_array = np.zeros(len(bonds), dtype=BondV2 if boltz2 else Bond)
    for bond, (atom1, atom2, bond_type) in zip(bond_array, bonds):
        bond["atom_1"] = atom1
        bond["atom_2"] = atom2
        bond["type"] = const.bond_type_ids[bond_type]
    fields = dict(
        atoms=atoms, bonds=bond_array, residues=residues, chains=chains,
        interfaces=np.zeros(0, dtype=Interface), mask=np.ones(len(chains), dtype=bool),
    )
    if boltz2:
        ensemble = np.zeros(1, dtype=Ensemble)
        ensemble[0]["atom_num"] = len(atoms)
        coords = np.zeros(len(atoms), dtype=Coords)
        coords["coords"] = atoms["coords"]
        return StructureV2(**fields, coords=coords, ensemble=ensemble)
    return Structure(**fields, connections=np.zeros(0, dtype=Connection))


@pytest.mark.parametrize("boltz2", [False, True], ids=["boltz1", "boltz2"])
class TestSparseAtomConnectivity:
    """Source indices must resolve to emitted serials, including TER gaps (#719)."""

    def test_missing_atoms_and_endpoints(self, boltz2):
        structure = _connectivity_structure(boltz2, bonds=[
            (0, 2, "SINGLE"), (2, 3, "DOUBLE"),
            (0, 1, "SINGLE"), (1, 3, "TRIPLE"),
        ])
        structure.atoms[1]["is_present"] = False
        assert _conect_lines(to_pdb(structure, boltz2=boltz2)) == [
            "CONECT    1    2", "CONECT    2    3    3",
        ]

    def test_present_but_unwritten_endpoints(self, boltz2):
        structure = _connectivity_structure(boltz2, bonds=[
            (0, 1, "SINGLE"), (1, 3, "SINGLE"), (1, 2, "TRIPLE"),
        ])
        # Atoms 0 and 3 exist but are outside the exported residue's atom span.
        structure.residues[0]["atom_idx"] = 1
        structure.residues[0]["atom_num"] = 2
        if not boltz2:
            connections = np.zeros(3, dtype=Connection)
            connections["atom_1"] = [0, 1, 2]
            connections["atom_2"] = [1, 3, 1]
            structure = replace(structure, connections=connections)
        expected = ["CONECT    1    2    2    2"]
        if not boltz2:
            expected.append("CONECT    2    1")
        assert _conect_lines(to_pdb(structure, boltz2=boltz2)) == expected

    def test_multiple_chains_aliases_and_ter_serial_shifts(self, boltz2):
        structure = _connectivity_structure(boltz2, (3, 3, 3), bonds=[
            (0, 2, "DOUBLE"), (3, 5, "TRIPLE"), (6, 8, "SINGLE"),
            (2, 3, "SINGLE"),
        ])
        structure.atoms["is_present"][[1, 4, 7]] = False
        structure.chains["asym_id"] = [9, 0, 40]
        structure.chains["name"] = ["P1", "A", "L1"]
        structure.chains[1]["mol_type"] = const.chain_type_ids["NONPOLYMER"]
        structure.residues[1]["name"] = "ATP"
        if not boltz2:
            connections = np.zeros(3, dtype=Connection)
            connections["atom_1"] = [2, 1, 3]
            connections["atom_2"] = [3, 3, 4]
            structure = replace(structure, connections=connections)
        with pytest.warns(UserWarning, match="P1 -> B, L1 -> C"):
            lines = to_pdb(structure, boltz2=boltz2).splitlines()
        assert all(len(line) == 80 for line in lines)
        records = [line for line in lines if line.startswith(("ATOM", "HETATM", "TER"))]
        assert [(line[:6].strip(), int(line[6:11]), line[21]) for line in records] == [
            ("ATOM", 1, "B"), ("ATOM", 2, "B"), ("TER", 3, "B"),
            ("HETATM", 4, "A"), ("HETATM", 5, "A"), ("TER", 6, "A"),
            ("ATOM", 7, "C"), ("ATOM", 8, "C"),
        ]
        assert [(line[17:20], int(line[22:26])) for line in records if line.startswith("TER")] == [
            ("ALA", 1), ("ATP", 2),
        ]
        expected = [
            "CONECT    1    2    2", "CONECT    4    5    5    5",
            "CONECT    7    8", "CONECT    2    4",
        ]
        if not boltz2:
            expected.append("CONECT    2    4")
        assert _conect_lines("\n".join(lines)) == expected

    @pytest.mark.parametrize("empty_kind", ["no_residues", "no_atoms", "missing_atoms"])
    @pytest.mark.parametrize("empty_chain", [0, 1, 2])
    def test_empty_chains_do_not_emit_stale_ter(self, boltz2, empty_kind, empty_chain):
        structure = _connectivity_structure(boltz2, (1, 1, 1))
        if empty_kind == "no_residues":
            structure.chains[empty_chain]["res_num"] = 0
        elif empty_kind == "no_atoms":
            structure.residues[empty_chain]["atom_num"] = 0
        else:
            structure.atoms[empty_chain]["is_present"] = False
        lines = to_pdb(structure, boltz2=boltz2).splitlines()
        records = [line for line in lines if line.startswith(("ATOM", "TER"))]
        expected = []
        serial = 1
        for i, tag in enumerate("ABC"):
            if i == empty_chain:
                continue
            expected.append(("ATOM", serial, tag))
            serial += 1
            if i < 2:
                expected.append(("TER", serial, tag))
                serial += 1
        assert [(line[:6].strip(), int(line[6:11]), line[21]) for line in records] == expected

    @pytest.mark.parametrize("atom_counts", [(), (0,), (0, 0), (2,), (2, 2)])
    def test_no_written_atoms(self, boltz2, atom_counts):
        structure = _connectivity_structure(boltz2, atom_counts)
        structure.atoms["is_present"] = False
        assert to_pdb(structure, boltz2=boltz2) == f"{'END':<80}\n"

    def test_missing_residues_keep_confidence_and_ter_identity(self, boltz2):
        structure = _connectivity_structure(boltz2, (1, 1, 1, 1))
        # Export the first three residues as one chain; only its middle residue emits.
        chains = structure.chains[[0, 3]].copy()
        chains[0]["res_num"] = 3
        chains[0]["atom_num"] = 3
        structure = replace(structure, chains=chains)
        structure.atoms["is_present"][[0, 2]] = False
        structure.residues[1]["name"] = "GLY"
        lines = to_pdb(structure, boltz2=boltz2, plddts=torch.tensor([.1, .2, .3, .4])).splitlines()
        atoms = [line for line in lines if line.startswith("ATOM")]
        assert [float(line[60:66]) for line in atoms] == [20.0, 40.0]
        ter = [line for line in lines if line.startswith("TER")]
        assert [(line[17:20], line[21], int(line[22:26])) for line in ter] == [("GLY", "A", 2)]


class TestConectBondOrder:
    """CONECT repeats a bonded partner to mark double and triple bonds (upstream #196)."""

    def _ligand_with_bonds(self, bond_types):
        """A ligand with one bond per type, each between atom 0 and atom i+1."""
        n = len(bond_types) + 1
        structure = _make_boltz2_ligand_structure(atom_name_str="C1", element=6)
        atoms = np.zeros(n, dtype=AtomV2)
        for i in range(n):
            atoms[i]["name"] = f"C{i + 1}"
            atoms[i]["element"] = 6
            atoms[i]["coords"] = (float(i), 0.0, 0.0)
            atoms[i]["is_present"] = True
        bonds = np.zeros(len(bond_types), dtype=BondV2)
        for i, bond_type in enumerate(bond_types):
            bonds[i]["atom_1"] = 0
            bonds[i]["atom_2"] = i + 1
            bonds[i]["type"] = const.bond_type_ids[bond_type]
        structure.residues[0]["atom_num"] = n
        structure.chains[0]["atom_num"] = n
        structure.ensemble[0]["atom_num"] = n
        return StructureV2(
            atoms=atoms,
            bonds=bonds,
            residues=structure.residues,
            chains=structure.chains,
            interfaces=structure.interfaces,
            mask=structure.mask,
            coords=np.array([(a["coords"],) for a in atoms], dtype=Coords),
            ensemble=structure.ensemble,
        )

    def test_partner_repeated_by_bond_order(self):
        structure = self._ligand_with_bonds(
            ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER", "COVALENT"]
        )
        lines = _conect_lines(to_pdb(structure, boltz2=True))
        assert lines == [
            "CONECT    1    2",
            "CONECT    1    3    3",
            "CONECT    1    4    4    4",
            "CONECT    1    5",
            "CONECT    1    6",
            "CONECT    1    7",
        ]

    def test_conect_lines_stay_80_columns(self):
        structure = self._ligand_with_bonds(["TRIPLE"])
        for line in to_pdb(structure, boltz2=True).split("\n"):
            if line.startswith("CONECT"):
                assert len(line) == 80

    def test_boltz1_connections_without_bond_type_are_written_once(self, minimal_structure):
        """Connection records have no type field; reading one must not crash."""
        import dataclasses

        connection = np.zeros(1, dtype=Connection)
        connection[0]["atom_1"] = 0
        connection[0]["atom_2"] = 5
        structure = dataclasses.replace(minimal_structure, connections=connection)
        lines = _conect_lines(to_pdb(structure))
        assert lines[-1] == "CONECT    1    6"


class TestPdbChainIds:
    """PDB has one column for the chain ID; longer names used to shift columns.

    Reported against upstream in jwohlwend/boltz#210.
    """

    def test_single_character_names_are_kept(self):
        from boltz.data.write.pdb import _pdb_chain_ids

        assert _pdb_chain_ids(["A", "B", "Z"]) == ["A", "B", "Z"]

    def test_long_names_get_unused_characters_in_chain_order(self):
        from boltz.data.write.pdb import _pdb_chain_ids

        assert _pdb_chain_ids(["A", "L1", "B", "L2"]) == ["A", "C", "B", "D"]

    def test_truncation_collisions_are_avoided(self):
        """Truncating L1 and L2 would give both chains the ID L."""
        from boltz.data.write.pdb import _pdb_chain_ids

        ids = _pdb_chain_ids(["L1", "L2", "L"])
        assert ids[2] == "L"
        assert len(set(ids)) == 3

    def test_duplicate_names_get_distinct_ids(self):
        from boltz.data.write.pdb import _pdb_chain_ids

        assert len(set(_pdb_chain_ids(["A", "A"]))) == 2

    def test_too_many_chains_raises_clear_error(self):
        from boltz.data.write.pdb import _pdb_chain_ids

        with pytest.raises(ValueError, match="mmcif"):
            _pdb_chain_ids([f"C{i}" for i in range(63)])

    def test_multi_character_chain_keeps_columns_aligned(self, minimal_structure_with_ligand):
        import dataclasses

        chains = minimal_structure_with_ligand.chains.copy()
        chains[0]["name"] = "P1"
        chains[1]["name"] = "L1"
        structure = dataclasses.replace(minimal_structure_with_ligand, chains=chains)

        with pytest.warns(UserWarning, match="P1 -> A, L1 -> B"):
            pdb_str = to_pdb(structure)

        lines = [l for l in pdb_str.split("\n") if l.strip()]
        assert all(len(l) == 80 for l in lines)

        records = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
        assert {l[21] for l in records if l.startswith("ATOM")} == {"A"}
        assert {l[21] for l in records if l.startswith("HETATM")} == {"B"}
        for line, atom in zip(records, structure.atoms):
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            assert xyz == pytest.approx(atom["coords"].tolist(), abs=1e-3)

        ter = [l for l in lines if l.startswith("TER")]
        assert [l[21] for l in ter] == ["A"]

    def test_no_warning_when_names_already_fit(self, minimal_structure_with_ligand):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            to_pdb(minimal_structure_with_ligand)
