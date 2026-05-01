import io
import logging
from collections.abc import Iterator
from typing import Optional

import ihm
import modelcif
import numpy as np
from gemmi import cif
from modelcif import Assembly, AsymUnit, Entity, System, dumper
from modelcif.model import AbInitioModel, Atom, ModelGroup
from rdkit import Chem
from torch import Tensor

from boltz.data import const
from boltz.data.types import Structure

logger = logging.getLogger(__name__)

_STRUCT_CONN_CATEGORY = "_struct_conn."
_STRUCT_CONN_COLUMNS = [
    "id",
    "conn_type_id",
    "ptnr1_label_asym_id",
    "ptnr1_label_comp_id",
    "ptnr1_label_seq_id",
    "ptnr1_label_atom_id",
    "ptnr1_auth_asym_id",
    "ptnr1_auth_comp_id",
    "ptnr1_auth_seq_id",
    "ptnr1_auth_atom_id",
    "ptnr2_label_asym_id",
    "ptnr2_label_comp_id",
    "ptnr2_label_seq_id",
    "ptnr2_label_atom_id",
    "ptnr2_auth_asym_id",
    "ptnr2_auth_comp_id",
    "ptnr2_auth_seq_id",
    "ptnr2_auth_atom_id",
    "pdbx_dist_value",
]


def _atom_name(atom, boltz2: bool) -> str:
    """Return an atom name for either Structure or StructureV2 atoms."""
    if boltz2:
        return str(atom["name"])
    return "".join(chr(c + 32) for c in atom["name"] if c != 0)


def _cross_residue_bonds(structure: Structure):
    """Yield cross-residue covalent bond-like rows from Structure/StructureV2."""
    bond_names = structure.bonds.dtype.names or ()
    covalent_type = const.bond_type_ids.get("COVALENT")

    if {"chain_1", "chain_2", "res_1", "res_2", "atom_1", "atom_2", "type"} <= set(
        bond_names
    ):
        for bond in structure.bonds:
            if (int(bond["res_1"]) == int(bond["res_2"])) and (
                int(bond["chain_1"]) == int(bond["chain_2"])
            ):
                continue
            if covalent_type is not None and int(bond["type"]) != covalent_type:
                continue
            yield bond
        return

    if not hasattr(structure, "connections"):
        return

    connection_names = structure.connections.dtype.names or ()
    if "type" in connection_names:
        msg = (
            "Structure.connections now includes a type field; update "
            "_cross_residue_bonds to filter connection types explicitly."
        )
        raise ValueError(msg)

    # Structure.Connection has no type today; parse_boltz_schema only stores
    # user-provided covalent bond constraints there.
    for connection in structure.connections:
        if (int(connection["res_1"]) == int(connection["res_2"])) and (
            int(connection["chain_1"]) == int(connection["chain_2"])
        ):
            continue
        yield connection


def _struct_conn_rows(structure: Structure, boltz2: bool) -> list[tuple]:
    """Create struct_conn rows for cross-residue covalent bonds."""
    rows = []
    chain_by_id = {int(chain["asym_id"]): chain for chain in structure.chains}

    for bond in _cross_residue_bonds(structure):
        chain_1 = chain_by_id.get(int(bond["chain_1"]))
        chain_2 = chain_by_id.get(int(bond["chain_2"]))
        if chain_1 is None or chain_2 is None:
            continue

        res_1 = structure.residues[int(bond["res_1"])]
        res_2 = structure.residues[int(bond["res_2"])]
        atom_1 = structure.atoms[int(bond["atom_1"])]
        atom_2 = structure.atoms[int(bond["atom_2"])]
        if not atom_1["is_present"] or not atom_2["is_present"]:
            logger.warning(
                "Skipping struct_conn record because a covalent bond endpoint "
                "atom is not present."
            )
            continue

        atom_name_1 = _atom_name(atom_1, boltz2)
        atom_name_2 = _atom_name(atom_2, boltz2)
        dist = float(np.linalg.norm(atom_1["coords"] - atom_2["coords"]))
        rows.append(
            (
                f"covale{len(rows) + 1}",
                "covale",
                chain_1["name"],
                res_1["name"],
                int(res_1["res_idx"]) + 1,
                atom_name_1,
                chain_1["name"],
                res_1["name"],
                int(res_1["res_idx"]) + 1,
                atom_name_1,
                chain_2["name"],
                res_2["name"],
                int(res_2["res_idx"]) + 1,
                atom_name_2,
                chain_2["name"],
                res_2["name"],
                int(res_2["res_idx"]) + 1,
                atom_name_2,
                f"{dist:.3f}",
            )
        )

    return rows


# Stable helper tested in tests/data/write/test_mmcif_writer.py.
def _add_struct_conn_records(mmcif: str, rows: list[tuple]) -> str:
    """Add struct_conn rows to an existing mmCIF document."""
    if not rows:
        return mmcif

    data_blocks, has_struct_conn = _scan_top_level(mmcif)
    if data_blocks != 1:
        msg = (
            "Expected exactly one mmCIF data block when adding struct_conn "
            f"records, found {data_blocks}."
        )
        raise ValueError(msg)

    if not has_struct_conn:
        separator = "" if mmcif.endswith("\n") else "\n"
        return mmcif + separator + _format_struct_conn_loop(rows)

    doc = cif.read_string(mmcif)
    block = doc[0]
    item = block.find_loop_item("_struct_conn.id")
    if item is None or item.loop is None:
        loop = block.init_loop(_STRUCT_CONN_CATEGORY, _STRUCT_CONN_COLUMNS)
    else:
        loop = item.loop
        missing_columns = [
            f"{_STRUCT_CONN_CATEGORY}{column}"
            for column in _STRUCT_CONN_COLUMNS
            if f"{_STRUCT_CONN_CATEGORY}{column}" not in loop.tags
        ]
        if missing_columns:
            loop.add_columns(missing_columns, "?")

    rows = _with_unique_struct_conn_ids(rows, loop)
    for row in rows:
        row_values = {
            f"{_STRUCT_CONN_CATEGORY}{column}": cif.quote(str(value))
            for column, value in zip(_STRUCT_CONN_COLUMNS, row)
        }
        loop.add_row([row_values.get(tag, "?") for tag in loop.tags])
    return doc.as_string()


def _scan_top_level(mmcif: str) -> tuple[int, bool]:
    """Return top-level data-block count and whether struct_conn is present."""
    data_blocks = 0
    has_struct_conn = False
    in_text_field = False

    for line in mmcif.splitlines():
        if line.startswith(";"):
            in_text_field = not in_text_field
            continue
        if not in_text_field:
            lowered = line.lower()
            if lowered.startswith("data_"):
                data_blocks += 1
            elif lowered.startswith(_STRUCT_CONN_CATEGORY):
                has_struct_conn = True

    if in_text_field:
        msg = "Malformed mmCIF contains an unterminated text field."
        raise ValueError(msg)

    return data_blocks, has_struct_conn


def _format_struct_conn_loop(rows: list[tuple]) -> str:
    """Format a standalone struct_conn loop."""
    lines = [
        "loop_",
        *(f"{_STRUCT_CONN_CATEGORY}{column}" for column in _STRUCT_CONN_COLUMNS),
    ]
    lines.extend(" ".join(cif.quote(str(value)) for value in row) for row in rows)
    lines.append("#")
    return "\n".join(lines) + "\n"


def _with_unique_struct_conn_ids(rows: list[tuple], loop) -> list[tuple]:
    """Renumber covale IDs to avoid collisions with existing struct_conn rows."""
    id_tag = f"{_STRUCT_CONN_CATEGORY}id"
    if id_tag not in loop.tags:
        return rows

    id_col = loop.tags.index(id_tag)
    existing_ids = {
        cif.as_string(loop[row_idx, id_col]) for row_idx in range(loop.length())
    }
    next_covale = 1
    for existing_id in existing_ids:
        if existing_id.startswith("covale") and existing_id[6:].isdigit():
            next_covale = max(next_covale, int(existing_id[6:]) + 1)

    unique_rows = []
    for row in rows:
        row = list(row)
        if str(row[0]).startswith("covale"):
            while f"covale{next_covale}" in existing_ids:
                next_covale += 1
            row[0] = f"covale{next_covale}"
            existing_ids.add(row[0])
            next_covale += 1
        unique_rows.append(tuple(row))
    return unique_rows


def to_mmcif(
    structure: Structure,
    plddts: Optional[Tensor] = None,
    boltz2: bool = False,
) -> str:  # noqa: C901, PLR0915, PLR0912
    """Write a structure into an MMCIF file.

    Parameters
    ----------
    structure : Structure
        The input Boltz structure.
    plddts : Optional[Tensor]
        Per-token confidence scores to write as B-factors and ModelCIF QA metrics.
    boltz2 : bool
        Whether the input uses Boltz-2 atom-name encoding.

    Returns
    -------
    str
        The output MMCIF file, including cross-residue covalent bonds as
        ``_struct_conn`` records when present.

    """
    system = System()

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    # Map entities to chain_ids
    entity_to_chains = {}
    entity_to_moltype = {}

    for chain in structure.chains:
        entity_id = chain["entity_id"]
        mol_type = chain["mol_type"]
        entity_to_chains.setdefault(entity_id, []).append(chain)
        entity_to_moltype[entity_id] = mol_type

    # Map entities to sequences
    sequences = {}
    for entity in entity_to_chains:
        # Get the first chain
        chain = entity_to_chains[entity][0]

        # Get the sequence
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        residues = structure.residues[res_start:res_end]
        sequence = [str(res["name"]) for res in residues]
        sequences[entity] = sequence

    # Create entity objects
    entities_map = {}
    for entity, sequence in sequences.items():
        mol_type = entity_to_moltype[entity]

        if mol_type == const.chain_type_ids["PROTEIN"]:
            alphabet = ihm.LPeptideAlphabet()
            chem_comp = lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")  # noqa: E731
        elif mol_type == const.chain_type_ids["DNA"]:
            alphabet = ihm.DNAAlphabet()
            chem_comp = lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif mol_type == const.chain_type_ids["RNA"]:
            alphabet = ihm.RNAAlphabet()
            chem_comp = lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif len(sequence) > 1:
            alphabet = {}
            chem_comp = lambda x: ihm.SaccharideChemComp(id=x)  # noqa: E731
        else:
            alphabet = {}
            chem_comp = lambda x: ihm.NonPolymerChemComp(id=x)  # noqa: E731

        seq = [
            alphabet[item] if item in alphabet else chem_comp(item)
            for item in sequence
        ]
        model_e = Entity(seq)

        for chain in entity_to_chains[entity]:
            chain_idx = chain["asym_id"]
            entities_map[chain_idx] = model_e

    # We don't assume that symmetry is perfect, so we dump everything
    # into the asymmetric unit, and produce just a single assembly
    asym_unit_map = {}
    for chain in structure.chains:
        # Define the model assembly
        chain_idx = chain["asym_id"]
        chain_tag = str(chain["name"])
        entity = entities_map[chain_idx]
        if entity.type == "water":
            asym = ihm.WaterAsymUnit(
                entity,
                1,
                details="Model subunit %s" % chain_tag,
                id=chain_tag,
            )
        else:
            asym = AsymUnit(
                entity,
                details="Model subunit %s" % chain_tag,
                id=chain_tag,
            )
        asym_unit_map[chain_idx] = asym
    modeled_assembly = Assembly(asym_unit_map.values(), name="Modeled assembly")

    class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Predicted lddt"

    class _MyModel(AbInitioModel):
        def get_atoms(self) -> Iterator[Atom]:
            # Index into plddt tensor for current residue.
            res_num = 0
            # Tracks non-ligand plddt tensor indices,
            # Initializing to -1 handles case where ligand is resnum 0
            prev_polymer_resnum = -1
            # Tracks ligand indices.
            ligand_index_offset = 0

            # Add all atom sites.
            for chain in structure.chains:
                # We rename the chains in alphabetical order
                het = chain["mol_type"] == const.chain_type_ids["NONPOLYMER"]
                chain_idx = chain["asym_id"]
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]

                record_type = (
                    "ATOM"
                    if chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    else "HETATM"
                )

                residues = structure.residues[res_start:res_end]
                for residue in residues:
                    atom_start = residue["atom_idx"]
                    atom_end = residue["atom_idx"] + residue["atom_num"]
                    atoms = structure.atoms[atom_start:atom_end]
                    atom_coords = atoms["coords"]
                    for i, atom in enumerate(atoms):
                        # This should not happen on predictions, but just in case.
                        if not atom["is_present"]:
                            continue

                        if boltz2:
                            atom_name = str(atom["name"])
                            element = periodic_table.GetElementSymbol(
                                atom["element"].item()
                            )
                        else:
                            atom_name = atom["name"]
                            atom_name = [chr(c + 32) for c in atom_name if c != 0]
                            atom_name = "".join(atom_name)
                            element = periodic_table.GetElementSymbol(
                                atom["element"].item()
                            )
                        element = element.upper()
                        residue_index = residue["res_idx"] + 1
                        pos = atom_coords[i]

                        if record_type != "HETATM":
                            # The current residue plddt is stored at the res_num index unless a ligand has previouly been added.
                            biso = (
                                100.00
                                if plddts is None
                                else round(
                                    plddts[res_num + ligand_index_offset].item() * 100,
                                    3,
                                )
                            )
                            prev_polymer_resnum = res_num
                        else:
                            # If not a polymer resnum, we can get index into plddts by adding offset relative to previous polymer resnum.
                            ligand_index_offset += 1
                            biso = (
                                100.00
                                if plddts is None
                                else round(
                                    plddts[
                                        prev_polymer_resnum + ligand_index_offset
                                    ].item()
                                    * 100,
                                    3,
                                )
                            )

                        yield Atom(
                            asym_unit=asym_unit_map[chain_idx],
                            type_symbol=element,
                            seq_id=residue_index,
                            atom_id=atom_name,
                            x=f"{pos[0]:.5f}",
                            y=f"{pos[1]:.5f}",
                            z=f"{pos[2]:.5f}",
                            het=het,
                            biso=biso,
                            occupancy=1,
                        )

                    if record_type != "HETATM":
                        res_num += 1

        def add_plddt(self, plddts):
            res_num = 0
            prev_polymer_resnum = (
                -1
            )  # -1 handles case where ligand is the first residue
            ligand_index_offset = 0
            for chain in structure.chains:
                chain_idx = chain["asym_id"]
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]
                residues = structure.residues[res_start:res_end]

                record_type = (
                    "ATOM"
                    if chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    else "HETATM"
                )

                # We rename the chains in alphabetical order
                for residue in residues:
                    residue_idx = residue["res_idx"] + 1

                    if record_type != "HETATM":
                        # The current residue plddt is stored at the res_num index unless a ligand has previouly been added.
                        self.qa_metrics.append(
                            _LocalPLDDT(
                                asym_unit_map[chain_idx].residue(residue_idx),
                                round(
                                    plddts[res_num + ligand_index_offset].item() * 100,
                                    3,
                                ),
                            )
                        )
                        prev_polymer_resnum = res_num
                    else:
                        # If not a polymer resnum, we can get index into plddts by adding offset relative to previous polymer resnum.
                        self.qa_metrics.append(
                            _LocalPLDDT(
                                asym_unit_map[chain_idx].residue(residue_idx),
                                round(
                                    plddts[
                                        prev_polymer_resnum
                                        + ligand_index_offset
                                        + 1 : prev_polymer_resnum
                                        + ligand_index_offset
                                        + residue["atom_num"]
                                        + 1
                                    ]
                                    .mean()
                                    .item()
                                    * 100,
                                    2,
                                ),
                            )
                        )
                        ligand_index_offset += residue["atom_num"]

                    if record_type != "HETATM":
                        res_num += 1

    # Add the model and modeling protocol to the file and write them out:
    model = _MyModel(assembly=modeled_assembly, name="Model")
    if plddts is not None:
        model.add_plddt(plddts)

    model_group = ModelGroup([model], name="All models")
    system.model_groups.append(model_group)
    ihm.dumper.set_line_wrap(False)

    fh = io.StringIO()
    dumper.write(fh, [system])
    return _add_struct_conn_records(
        fh.getvalue(),
        _struct_conn_rows(structure, boltz2=boltz2),
    )
