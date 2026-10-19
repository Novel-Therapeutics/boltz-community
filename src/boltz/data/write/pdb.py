import warnings
from typing import Optional

from rdkit import Chem
from torch import Tensor

from boltz.data import const
from boltz.data.types import Structure


_CONECT_REPEATS = {
    const.bond_type_ids["DOUBLE"]: 2,
    const.bond_type_ids["TRIPLE"]: 3,
}


def _conect_bond_order(bond) -> int:
    """Return how many times to list a bond's partner in its CONECT record.

    Listing a partner twice (double) or three times (triple) is the common,
    if unofficial, convention RDKit, Open Babel and PyMOL use for bond order.
    Covalent connection records (``Connection``) carry no bond type and are
    written once, like single, aromatic and other bonds.
    """
    if "type" not in (bond.dtype.names or ()):
        return 1
    return _CONECT_REPEATS.get(int(bond["type"]), 1)


_PDB_CHAIN_ID_POOL = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)


def _pdb_chain_ids(names: list[str]) -> list[str]:
    """Assign each chain, in order, a unique one-character PDB chain ID.

    The PDB format has a single column for the chain ID, so a longer name such
    as ``L1`` shifts every later column and corrupts the coordinates. Names that
    are already one character and unique keep their ID; every other chain gets
    the first unused character, in chain order. Raises ``ValueError`` when the
    62 available IDs run out, since PDB cannot represent that many chains.
    """
    single = [n for n in names if len(n) == 1]
    keep = {n for n in single if single.count(n) == 1}
    pool = (c for c in _PDB_CHAIN_ID_POOL if c not in keep)
    ids = []
    for name in names:
        if name in keep:
            ids.append(name)
            continue
        try:
            ids.append(next(pool))
        except StopIteration:
            msg = (
                f"Cannot write {len(names)} chains to PDB, which allows at most "
                f"{len(_PDB_CHAIN_ID_POOL)} one-character chain IDs. "
                "Use --output_format mmcif instead."
            )
            raise ValueError(msg) from None
    return ids


def to_pdb(
    structure: Structure,
    plddts: Optional[Tensor] = None,
    boltz2: bool = False,
) -> str:  # noqa: PLR0915
    """Write a structure into a PDB file.

    Parameters
    ----------
    structure : Structure
        The input structure

    Returns
    -------
    str
        the output PDB file

    """
    pdb_lines = []

    atom_index = 1
    atom_reindex_ter = []

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    # Index into plddt tensor for current residue.
    res_num = 0
    # Tracks non-ligand plddt tensor indices,
    # Initializing to -1 handles case where ligand is resnum 0
    prev_polymer_resnum = -1
    # Tracks ligand indices.
    ligand_index_offset = 0

    # Add all atom sites.
    chain_names = [str(chain["name"]) for chain in structure.chains]
    pdb_chain_ids = _pdb_chain_ids(chain_names)
    renamed = [(n, c) for n, c in zip(chain_names, pdb_chain_ids) if n != c]
    if renamed:
        details = ", ".join(f"{n} -> {c}" for n, c in renamed)
        warnings.warn(
            "PDB chain IDs are one character, so these chains were renamed in "
            f"the PDB output: {details}. mmCIF output keeps the original names.",
            stacklevel=2,
        )

    for chain, chain_tag in zip(structure.chains, pdb_chain_ids):
        # We rename the chains in alphabetical order
        chain_idx = chain["asym_id"]

        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]

        residues = structure.residues[res_start:res_end]
        for residue in residues:
            res_name = str(residue["name"])
            atom_start = residue["atom_idx"]
            atom_end = residue["atom_idx"] + residue["atom_num"]
            atoms = structure.atoms[atom_start:atom_end]
            atom_coords = atoms["coords"]
            for i, atom in enumerate(atoms):
                # This should not happen on predictions, but just in case.
                if not atom["is_present"]:
                    continue

                record_type = (
                    "ATOM"
                    if chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    else "HETATM"
                )
                name = str(atom["name"])
                if boltz2:
                    element = periodic_table.GetElementSymbol(
                        atom["element"].item()
                    )
                else:
                    atom_name = atom["name"]
                    atom_name = [chr(c + 32) for c in atom_name if c != 0]
                    atom_name = "".join(atom_name)
                    name = atom_name
                    element = periodic_table.GetElementSymbol(atom["element"].item())

                name = name if len(name) == 4 else f" {name}"  # noqa: PLR2004
                alt_loc = ""
                insertion_code = ""
                occupancy = 1.00
                element = element.upper()
                charge = ""
                residue_index = residue["res_idx"] + 1
                pos = atom_coords[i]
                res_name_3 = str(residue["name"][:3])

                if record_type != "HETATM":
                    # The current residue plddt is stored at the res_num index unless a ligand has previouly been added.
                    b_factor = (
                        100.00
                        if plddts is None
                        else round(
                            plddts[res_num + ligand_index_offset].item() * 100, 2
                        )
                    )
                    prev_polymer_resnum = res_num
                else:
                    # If not a polymer resnum, we can get index into plddts by adding offset relative to previous polymer resnum.
                    ligand_index_offset += 1
                    b_factor = (
                        100.00
                        if plddts is None
                        else round(
                            plddts[prev_polymer_resnum + ligand_index_offset].item()
                            * 100,
                            2,
                        )
                    )

                # PDB is a columnar format, every space matters here!
                # Format in a single pass: 80 chars wide per PDB spec.
                atom_line = (
                    f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                    f"{res_name_3:>3} {chain_tag:>1}"
                    f"{residue_index:>4}{insertion_code:>1}   "
                    f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                    f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                    f"{element:>2}{charge:<2}"
                )
                pdb_lines.append(atom_line)
                atom_reindex_ter.append(atom_index)
                atom_index += 1

            if record_type != "HETATM":
                res_num += 1

        should_terminate = chain_idx < (len(structure.chains) - 1)
        if should_terminate:
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      "
                f"{res_name_3:>3} "
                f"{chain_tag:>1}{residue_index:>4}"
            )
            pdb_lines.append(f"{chain_termination_line:<80}")
            atom_index += 1

    # Dump CONECT records.
    all_bonds = list(structure.bonds)
    if hasattr(structure, "connections"):
        all_bonds.extend(list(structure.connections))

    for bond in all_bonds:
        atom1 = structure.atoms[bond["atom_1"]]
        atom2 = structure.atoms[bond["atom_2"]]
        if not atom1["is_present"] or not atom2["is_present"]:
            continue
        atom1_idx = atom_reindex_ter[bond["atom_1"]]
        atom2_idx = atom_reindex_ter[bond["atom_2"]]
        conect_line = f"CONECT{atom1_idx:>5}" + f"{atom2_idx:>5}" * _conect_bond_order(bond)
        pdb_lines.append(f"{conect_line:<80}")

    pdb_lines.append(f"{'END':<80}")
    pdb_lines.append("")
    return "\n".join(pdb_lines)
