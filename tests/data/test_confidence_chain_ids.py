"""Confidence indices remain traceable to original input chain IDs."""

import json
from dataclasses import asdict, replace

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
    ChainInfo,
    Connection,
    Coords,
    Ensemble,
    Input,
    Interface,
    Record,
    Residue,
    Structure,
    StructureInfo,
    StructureV2,
)
from boltz.data.write.writer import BoltzWriter
from boltz.model.layers.structure_checks import CHECK_KEYS

SCALARS = [
    "confidence_score",
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
]


def write_case(
    tmp_path,
    boltz2,
    output_format,
    mask,
    labels=None,
    confidence=True,
    batch_size=1,
    native_labels=None,
    use_tokenizer=False,
):
    count = len(mask)
    if labels is None:
        labels = [chr(65 + i) for i in range(count)]
    atoms = np.zeros(count, dtype=AtomV2 if boltz2 else Atom)
    atoms["is_present"] = True
    atoms["element"] = 6
    if boltz2:
        atoms["name"] = "CA"
    else:
        atoms["name"] = (ord("C") - 32, ord("A") - 32, 0, 0)
    residues = np.zeros(count, dtype=Residue)
    chains = np.zeros(count, dtype=Chain)
    for i in range(count):
        residues[i]["name"] = "GLY"
        residues[i]["res_type"] = const.token_ids["GLY"]
        residues[i]["atom_idx"] = residues[i]["atom_center"] = residues[i][
            "atom_disto"
        ] = i
        residues[i]["atom_num"] = 1
        residues[i]["is_standard"] = residues[i]["is_present"] = True
        chains[i]["name"] = native_labels[i] if native_labels else chr(65 + i)
        chains[i]["mol_type"] = const.chain_type_ids["PROTEIN"]
        chains[i]["asym_id"] = i
        chains[i]["entity_id"] = 0  # Identical sequences are copies of one entity.
        chains[i]["res_idx"] = chains[i]["atom_idx"] = i
        chains[i]["res_num"] = chains[i]["atom_num"] = 1
    if use_tokenizer:
        # One-atom ligands exercise both real tokenizers without protein frames.
        chains["mol_type"] = const.chain_type_ids["NONPOLYMER"]
        residues["name"] = "LIG"
        residues["is_standard"] = False
    common = dict(
        atoms=atoms,
        residues=residues,
        chains=chains,
        interfaces=np.zeros(0, dtype=Interface),
        mask=np.array(mask),
    )
    if boltz2:
        structure = StructureV2(
            **common,
            bonds=np.zeros(0, dtype=BondV2),
            coords=np.zeros(count, dtype=Coords),
            ensemble=np.array([(0, count)], dtype=Ensemble),
        )
    else:
        structure = Structure(
            **common,
            bonds=np.zeros(0, dtype=Bond),
            connections=np.zeros(0, dtype=Connection),
        )
    source, output = tmp_path / "structures", tmp_path / "predictions"
    source.mkdir()
    np.savez_compressed(source / "toy.npz", **asdict(structure))
    record = Record(
        id="toy",
        structure=StructureInfo(num_chains=count),
        chains=[ChainInfo(i, label, 0, -1, -1, 1) for i, label in enumerate(labels)],
        interfaces=[],
    )
    records = [record]
    if batch_size == 2:
        np.savez_compressed(source / "other.npz", **asdict(structure))
        records.append(
            replace(
                record,
                id="other",
                chains=[
                    replace(chain, chain_name=f"other_{chain.chain_name}")
                    for chain in record.chains
                ],
            )
        )
    valid = sum(mask)
    score_ids = np.flatnonzero(mask).tolist()
    if use_tokenizer:
        from boltz.data.tokenize.boltz import BoltzTokenizer
        from boltz.data.tokenize.boltz2 import Boltz2Tokenizer

        tokenizer = Boltz2Tokenizer() if boltz2 else BoltzTokenizer()
        tokens = tokenizer.tokenize(Input(structure, {}, record)).tokens
        score_ids = [int(i) for i in np.unique(tokens["asym_id"])]
    # Two samples with reversed ranks catch accidental label/ranking coupling.
    samples = 2 * batch_size
    coords = torch.arange(samples * valid * 3, dtype=torch.float32).reshape(
        samples, valid, 3
    )
    if not confidence:
        coords = coords[:1]
    prediction = {
        "exception": False,
        "coords": coords,
        "masks": torch.ones(batch_size, valid),
    }
    if confidence:
        values = torch.tensor([0.25, 0.75, 0.8, 0.1][:samples])
        prediction.update({key: values.clone() for key in SCALARS})
        prediction["plddt"] = torch.full((samples, valid), 0.9)
        prediction["pair_chains_iptm"] = {
            i: {j: torch.arange(samples) * 0.5 + i / 10 + j / 100 for j in score_ids}
            for i in score_ids
        }
        for suffix in ("", "_unweighted"):
            prediction[f"complex_pae{suffix}"] = values + 1
            prediction[f"complex_ipae{suffix}"] = values + 2
            prediction[f"chains_pae{suffix}"] = {i: values + i for i in score_ids}
            prediction[f"pair_chains_pae{suffix}"] = {
                i: {j: values + i + j for j in score_ids} for i in score_ids
            }
            # Undefined summaries must retain the fork's null serialization.
            prediction[f"pair_chains_pae{suffix}"][score_ids[0]][score_ids[0]] = (
                torch.full((samples,), float("nan"))
            )
        prediction.update({key: torch.arange(samples) for key in CHECK_KEYS})
        prediction["pae"] = (
            torch.arange(samples * valid * valid).reshape(samples, valid, valid).float()
        )
        prediction["pde"] = prediction["pae"] + 10
    writer = BoltzWriter(source, output, output_format=output_format, boltz2=boltz2)
    writer.write_on_batch_end(None, None, prediction, [], {"record": records}, 0, 0)
    return (
        output / "toy",
        prediction,
        [label for label, kept in zip(labels, mask) if kept],
    )


@pytest.mark.parametrize("boltz2", [False, True])
@pytest.mark.parametrize("output_format", ["pdb", "mmcif"])
@pytest.mark.parametrize("mask", [[True, True, True], [True, False, True]])
def test_confidence_maps_reindexed_chains_and_preserves_all_scores(
    tmp_path, boltz2, output_format, mask
):
    output, prediction, names = write_case(tmp_path, boltz2, output_format, mask)
    for sample, rank in [(0, 1), (1, 0)]:
        summary = json.loads((output / f"confidence_toy_model_{rank}.json").read_text())
        assert summary.pop("chain_ids") == {
            str(i): name for i, name in zip(np.flatnonzero(mask), names)
        }
        pairs = {
            str(i): {str(j): value[sample].item() for j, value in row.items()}
            for i, row in prediction["pair_chains_iptm"].items()
        }
        expected = {key: prediction[key][sample].item() for key in SCALARS}
        expected.update(
            chains_ptm={i: row[i] for i, row in pairs.items()}, pair_chains_iptm=pairs
        )
        expected.update(extra_summary(prediction, sample))
        assert summary == expected
        suffix = "cif" if output_format == "mmcif" else "pdb"
        assert (output / f"toy_model_{rank}.{suffix}").is_file()
        plddt = np.load(output / f"plddt_toy_model_{rank}.npz")["plddt"]
        assert np.array_equal(plddt, prediction["plddt"][sample].numpy())


def extra_summary(prediction, sample):
    """Expected fork summaries, including undefined PAE and structure checks."""

    def select(value):
        if isinstance(value, dict):
            return {str(key): select(item) for key, item in value.items()}
        scalar = value[sample].item()
        return None if np.isnan(scalar) else scalar

    expected = {
        key: select(value)
        for key, value in prediction.items()
        if key.startswith(
            ("complex_pae", "complex_ipae", "chains_pae", "pair_chains_pae")
        )
    }
    expected.update({key: int(prediction[key][sample]) for key in CHECK_KEYS})
    expected["structure_checks_passed"] = sample == 0
    return expected


@pytest.mark.parametrize("boltz2", [False, True])
@pytest.mark.parametrize("output_format", ["pdb", "mmcif"])
def test_batch_ranks_keep_labels_scores_and_coordinates_together(
    tmp_path, boltz2, output_format
):
    import gemmi

    output, prediction, _ = write_case(
        tmp_path,
        boltz2,
        output_format,
        [False, True, False, True],
        labels=["removed", "binder_alias", "removed2", "target_alias"],
        batch_size=2,
    )
    for record, prefix, samples in [
        ("toy", "", [(0, 1), (1, 0)]),
        ("other", "other_", [(2, 0), (3, 1)]),
    ]:
        for sample, rank in samples:
            folder = output.parent / record
            summary = json.loads(
                (folder / f"confidence_{record}_model_{rank}.json").read_text()
            )
            assert summary["chain_ids"] == {
                "1": f"{prefix}binder_alias",
                "3": f"{prefix}target_alias",
            }
            assert (
                summary["confidence_score"]
                == prediction["confidence_score"][sample].item()
            )
            assert summary["chains_ptm"] == {
                str(i): prediction["pair_chains_iptm"][i][i][sample].item()
                for i in (1, 3)
            }
            for key, value in extra_summary(prediction, sample).items():
                assert summary[key] == value
            for key in ("plddt", "pae", "pde"):
                with np.load(folder / f"{key}_{record}_model_{rank}.npz") as data:
                    np.testing.assert_array_equal(
                        data[key], prediction[key][sample].numpy()
                    )
            suffix = "pdb" if output_format == "pdb" else "cif"
            structure = gemmi.read_structure(
                str(folder / f"{record}_model_{rank}.{suffix}")
            )
            # Native names survive; metadata aliases never rename coordinate chains.
            assert [chain.name for chain in structure[0]] == ["B", "D"]
            actual_coords = [
                [atom.pos.x, atom.pos.y, atom.pos.z]
                for chain in structure[0]
                for residue in chain
                for atom in residue
            ]
            np.testing.assert_allclose(
                actual_coords, prediction["coords"][sample].numpy()
            )


@pytest.mark.parametrize(
    "labels", [["input_receptor_long", "input_ligand_long"], ["same", "same"], ["only"]]
)
def test_original_names_are_not_truncated_or_used_as_unique_keys(tmp_path, labels):
    output, _, _ = write_case(
        tmp_path, True, "mmcif", [True] * len(labels), labels=labels
    )
    summary = json.loads((output / "confidence_toy_model_0.json").read_text())
    assert summary["chain_ids"] == {str(i): name for i, name in enumerate(labels)}


def test_prediction_without_confidence_still_writes_structures(tmp_path):
    output, _, _ = write_case(tmp_path, True, "mmcif", [True, True], confidence=False)
    assert (output / "toy_model_0.cif").is_file()
    assert not list(output.glob("confidence_*.json"))


@pytest.mark.parametrize("boltz2", [False, True])
def test_pdb_aliases_do_not_replace_original_confidence_labels(tmp_path, boltz2):
    import gemmi

    labels = ["gone", "L1", "A", "L2"]
    with pytest.warns(UserWarning, match="L1 -> B, L2 -> C"):
        output, _, _ = write_case(
            tmp_path,
            boltz2,
            "pdb",
            [False, True, True, True],
            labels=labels,
            native_labels=labels,
        )
    summary = json.loads((output / "confidence_toy_model_0.json").read_text())
    assert summary["chain_ids"] == {"1": "L1", "2": "A", "3": "L2"}
    parsed = gemmi.read_structure(str(output / "toy_model_0.pdb"))
    assert [chain.name for chain in parsed[0]] == ["B", "A", "C"]


@pytest.mark.parametrize("boltz2", [False, True])
@pytest.mark.parametrize("output_format", ["pdb", "mmcif"])
@pytest.mark.parametrize("mask", [[True, False, True], [False, True, True]])
def test_confidence_labels_follow_actual_tokenizer_indices(
    tmp_path, boltz2, output_format, mask
):
    output, prediction, _ = write_case(
        tmp_path, boltz2, output_format, mask, use_tokenizer=True
    )
    summary = json.loads((output / "confidence_toy_model_0.json").read_text())
    assert summary["chain_ids"] == {
        str(i): chr(65 + i) for i, kept in enumerate(mask) if kept
    }
    assert set(summary["chain_ids"]) == set(summary["chains_ptm"])
    assert summary["chains_ptm"] == {
        str(i): row[i][1].item() for i, row in prediction["pair_chains_iptm"].items()
    }
