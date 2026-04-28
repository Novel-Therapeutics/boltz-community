from pathlib import Path

import numpy as np
import pytest
import torch

from boltz.data import const
from boltz.data.types import (
    AffinityInfo,
    AtomV2,
    BondV2,
    Chain,
    ChainInfo,
    Coords,
    Ensemble,
    Interface,
    Record,
    Residue,
    StructureInfo,
    StructureV2,
)
from boltz.data.write.writer import (
    BoltzAffinityWriter,
    BoltzWriter,
    _select_prediction_value,
)


def _make_minimal_structure_v2() -> StructureV2:
    atoms = np.zeros(1, dtype=AtomV2)
    atoms[0]["name"] = "C1"
    atoms[0]["element"] = 6
    atoms[0]["coords"] = [0.0, 0.0, 0.0]
    atoms[0]["is_present"] = True

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

    coords = np.array([([0.0, 0.0, 0.0],)], dtype=Coords)
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
        coords=coords,
        ensemble=ensemble,
    )


def _make_record(record_id: str, affinity: AffinityInfo | None = None) -> Record:
    return Record(
        id=record_id,
        structure=StructureInfo(),
        chains=[
            ChainInfo(
                chain_id=0,
                chain_name="A",
                mol_type=const.chain_type_ids["NONPOLYMER"],
                cluster_id=0,
                msa_id=-1,
                num_residues=1,
            )
        ],
        interfaces=[],
        affinity=affinity,
    )


def test_select_prediction_value_serializes_nan_as_null():
    summary = {
        0: torch.tensor([float("nan")]),
        1: {
            0: torch.tensor([1.5]),
            1: torch.tensor([2.5]),
        },
    }

    assert _select_prediction_value(summary, 0) == {
        0: None,
        1: {
            0: 1.5,
            1: 2.5,
        },
    }


def test_boltz_writer_splits_batched_predictions_by_record(tmp_path: Path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    out_dir.mkdir()

    structure = _make_minimal_structure_v2()
    structure.dump(data_dir / "record_a.npz")
    structure.dump(data_dir / "record_b.npz")

    writer = BoltzWriter(
        data_dir=str(data_dir),
        output_dir=str(out_dir),
        output_format="mmcif",
        boltz2=True,
    )

    prediction = {
        "exception": False,
        "coords": torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0]],
            ]
        ),
        "masks": torch.ones(2, 1, dtype=torch.bool),
        "plddt": torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
        "pae": torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]]),
        "pde": torch.tensor([[[10.0]], [[20.0]], [[30.0]], [[40.0]]]),
        "confidence_score": torch.tensor([0.2, 0.9, 0.8, 0.1]),
        "ptm": torch.tensor([0.2, 0.9, 0.8, 0.1]),
        "iptm": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        "ligand_iptm": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        "protein_iptm": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        "complex_plddt": torch.tensor([0.2, 0.9, 0.8, 0.1]),
        "complex_iplddt": torch.tensor([0.2, 0.9, 0.8, 0.1]),
        "complex_pde": torch.tensor([0.2, 0.9, 0.8, 0.1]),
        "complex_ipde": torch.tensor([0.2, 0.9, 0.8, 0.1]),
        "pair_chains_iptm": {
            0: {
                0: torch.tensor([0.2, 0.9, 0.8, 0.1]),
            }
        },
    }
    batch = {
        "record": [_make_record("record_a"), _make_record("record_b")],
    }

    writer.write_on_batch_end(
        trainer=None,
        pl_module=None,
        prediction=prediction,
        batch_indices=[],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    record_a_dir = out_dir / "record_a"
    record_b_dir = out_dir / "record_b"

    assert (record_a_dir / "record_a_model_0.cif").exists()
    assert (record_a_dir / "record_a_model_1.cif").exists()
    assert (record_b_dir / "record_b_model_0.cif").exists()
    assert (record_b_dir / "record_b_model_1.cif").exists()

    import json

    with (record_a_dir / "confidence_record_a_model_0.json").open() as f:
        conf_a_top = json.load(f)
    with (record_a_dir / "confidence_record_a_model_1.json").open() as f:
        conf_a_low = json.load(f)
    with (record_b_dir / "confidence_record_b_model_0.json").open() as f:
        conf_b_top = json.load(f)
    with (record_b_dir / "confidence_record_b_model_1.json").open() as f:
        conf_b_low = json.load(f)

    assert conf_a_top["confidence_score"] == pytest.approx(0.9)
    assert conf_a_low["confidence_score"] == pytest.approx(0.2)
    assert conf_b_top["confidence_score"] == pytest.approx(0.8)
    assert conf_b_low["confidence_score"] == pytest.approx(0.1)

    np.testing.assert_array_equal(
        np.load(record_a_dir / "plddt_record_a_model_0.npz")["plddt"],
        prediction["plddt"][1].numpy(),
    )
    np.testing.assert_array_equal(
        np.load(record_b_dir / "plddt_record_b_model_0.npz")["plddt"],
        prediction["plddt"][2].numpy(),
    )
    np.testing.assert_array_equal(
        np.load(record_a_dir / "pae_record_a_model_0.npz")["pae"],
        prediction["pae"][1].numpy(),
    )
    np.testing.assert_array_equal(
        np.load(record_b_dir / "pae_record_b_model_0.npz")["pae"],
        prediction["pae"][2].numpy(),
    )
    np.testing.assert_array_equal(
        np.load(record_a_dir / "pde_record_a_model_0.npz")["pde"],
        prediction["pde"][1].numpy(),
    )
    np.testing.assert_array_equal(
        np.load(record_b_dir / "pde_record_b_model_0.npz")["pde"],
        prediction["pde"][2].numpy(),
    )


def test_boltz_writer_preserves_single_record_embedding_shape(tmp_path: Path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    out_dir.mkdir()

    structure = _make_minimal_structure_v2()
    structure.dump(data_dir / "record_a.npz")

    writer = BoltzWriter(
        data_dir=str(data_dir),
        output_dir=str(out_dir),
        output_format="mmcif",
        boltz2=True,
        write_embeddings=True,
    )

    prediction = {
        "exception": False,
        "coords": torch.tensor([[[0.0, 0.0, 0.0]]]),
        "masks": torch.ones(1, 1, dtype=torch.bool),
        "s": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
        "z": torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2),
    }
    batch = {
        "record": [_make_record("record_a")],
    }

    writer.write_on_batch_end(
        trainer=None,
        pl_module=None,
        prediction=prediction,
        batch_indices=[],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    embeddings = np.load(out_dir / "record_a" / "embeddings_record_a.npz")
    assert embeddings["s"].shape == (1, 2, 3)
    assert embeddings["z"].shape == (1, 1, 2, 2)


def test_affinity_writer_uses_per_copy_output_name(tmp_path: Path):
    writer = BoltzAffinityWriter(
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
    )

    batch = {
        "record": [
            _make_record(
                "record_a",
                affinity=AffinityInfo(
                    chain_id=0,
                    mw=123.4,
                    chain_name="L2",
                    output_id="record_a_L2",
                ),
            )
        ],
    }
    prediction = {
        "exception": False,
        "affinity_pred_value": torch.tensor(1.25),
        "affinity_probability_binary": torch.tensor(0.75),
    }

    writer.write_on_batch_end(
        trainer=None,
        pl_module=None,
        prediction=prediction,
        batch_indices=[],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    import json

    path = tmp_path / "out" / "record_a" / "affinity_record_a_L2.json"
    with path.open() as f:
        affinity = json.load(f)

    assert affinity["binder_chain"] == "L2"
    assert affinity["affinity_pred_value"] == pytest.approx(1.25)
