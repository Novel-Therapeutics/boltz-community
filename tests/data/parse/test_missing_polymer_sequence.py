"""Template sequence fallbacks must preserve residue and ensemble coordinates."""

import warnings

import gemmi
import numpy as np
import pytest
from rdkit import Chem

from boltz.data import const
from boltz.data.parse.mmcif import parse_mmcif
from boltz.data.parse.pdb import parse_pdb


@pytest.fixture
def molecules():
    """Build named reference atoms without downloading CCD assets."""
    resources = {}
    for name in ["ALA", "GLY", "SER"]:
        mol = Chem.RWMol()
        for atom_name in const.ref_atoms[name]:
            atom = Chem.Atom(atom_name[0])
            atom.SetProp("name", atom_name)
            mol.AddAtom(atom)
        resources[name] = mol.GetMol()
    return resources


def make_structure(chains, full_sequences=None, models=1):
    """Build models with distinct coordinates and arbitrary author numbering."""
    structure = gemmi.Structure()
    for model_idx in range(models):
        model = gemmi.Model(str(model_idx + 1))
        for chain_idx, (chain_name, entries) in enumerate(chains.items()):
            chain = gemmi.Chain(chain_name)
            for residue_idx, (name, number, insertion) in enumerate(entries):
                residue = gemmi.Residue()
                residue.name = name
                residue.seqid = gemmi.SeqId(number, insertion)
                residue.het_flag = "A"
                for atom_idx, atom_name in enumerate(const.ref_atoms[name]):
                    atom = gemmi.Atom()
                    atom.name = atom_name
                    atom.element = gemmi.Element(atom_name[0])
                    atom.pos = gemmi.Position(
                        3 * residue_idx + atom_idx * 0.1,
                        model_idx + 0.2,
                        chain_idx + 0.3,
                    )
                    residue.add_atom(atom)
                chain.add_residue(residue)
            model.add_chain(chain)
        structure.add_model(model)
    structure.setup_entities()
    for chain_name, sequence in (full_sequences or {}).items():
        entity = structure.get_entity_of(structure[0][chain_name].get_polymer())
        entity.full_sequence = sequence
    return structure


def parse_template(tmp_path, source, file_format, molecules):
    """Exercise the real PDB conversion or mmCIF parser."""
    path = tmp_path / ("template." + file_format)
    if file_format == "pdb":
        path.write_text(source.make_pdb_string())
        parser = parse_pdb
    else:
        source.make_mmcif_document().write_file(str(path))
        parser = parse_mmcif
    return parser(
        str(path), mols=molecules, use_assembly=False, compute_interfaces=False
    )


def assert_model_coordinates(parsed, source):
    """Check every model's present atoms, including model and chain identity."""
    assert len(parsed.data.ensemble) == len(source)
    for model, ensemble in zip(source, parsed.data.ensemble):
        expected = np.array([a.pos.tolist() for c in model for r in c for a in r])
        start = ensemble["atom_coord_idx"]
        stop = start + ensemble["atom_num"]
        actual = parsed.data.coords["coords"][start:stop]
        np.testing.assert_allclose(
            actual[parsed.data.atoms["is_present"]], expected, atol=1e-6
        )


@pytest.mark.parametrize("file_format", ["pdb", "cif"])
@pytest.mark.parametrize("models", [1, 2])
@pytest.mark.parametrize(
    "numbering",
    [((1, " "), (2, " ")), ((42, " "), (47, " ")), ((42, "A"), (42, "B"))],
)
def test_missing_sequence_uses_observed_residues(
    tmp_path, molecules, file_format, models, numbering
):
    entries = [(name, n, ins) for name, (n, ins) in zip(["ALA", "GLY"], numbering)]
    source = make_structure(
        {"A": entries, "B": [("SER", 8, " "), ("ALA", 9, " ")]}, models=models
    )
    with pytest.warns(UserWarning, match="observed residues"):
        parsed = parse_template(tmp_path, source, file_format, molecules)

    if file_format == "pdb":
        assert parsed.sequences == {"A1": "AG", "B1": "SA"}
        assert parsed.data.chains["name"].tolist() == ["A1", "B1"]
    else:
        assert list(parsed.sequences.values()) == ["AG", "SA"]
    assert parsed.data.chains["res_num"].tolist() == [2, 2]
    assert parsed.data.residues["is_present"].all()
    assert parsed.data.atoms["is_present"].all()
    assert_model_coordinates(parsed, source)


@pytest.mark.parametrize("file_format", ["pdb", "cif"])
@pytest.mark.parametrize("models", [1, 2])
def test_existing_full_sequence_retains_unresolved_residue(
    tmp_path, molecules, file_format, models
):
    source = make_structure(
        {"A": [("ALA", 1, " "), ("GLY", 3, " ")]},
        full_sequences={"A": ["ALA", "SER", "GLY"]},
        models=models,
    )
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        parsed = parse_template(tmp_path, source, file_format, molecules)

    assert not [w for w in emitted if "observed residues" in str(w.message)]
    assert list(parsed.sequences.values()) == ["ASG"]
    assert parsed.data.residues["name"].tolist() == ["ALA", "SER", "GLY"]
    assert parsed.data.residues["is_present"].tolist() == [True, False, True]
    assert_model_coordinates(parsed, source)


def test_only_missing_chain_sequence_is_inferred(tmp_path, molecules):
    source = make_structure(
        {
            "A": [("ALA", 1, " "), ("GLY", 3, " ")],
            "B": [("SER", 42, "A"), ("ALA", 42, "B")],
        },
        full_sequences={"A": ["ALA", "SER", "GLY"]},
        models=2,
    )
    with pytest.warns(UserWarning, match="observed residues"):
        parsed = parse_template(tmp_path, source, "cif", molecules)

    assert list(parsed.sequences.values()) == ["ASG", "SA"]
    assert parsed.data.residues["is_present"].tolist() == [
        True,
        False,
        True,
        True,
        True,
    ]
    assert_model_coordinates(parsed, source)
