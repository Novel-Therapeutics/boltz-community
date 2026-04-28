from pathlib import Path
from dataclasses import replace

import click
import pytest

from boltz.data import const
from boltz.data.types import AffinityInfo, ChainInfo, Manifest, Record, StructureInfo
from boltz.main import expand_affinity_records, filter_inputs_affinity


def _make_affinity_record(record_id: str = "target") -> Record:
    return Record(
        id=record_id,
        structure=StructureInfo(),
        chains=[
            ChainInfo(
                chain_id=0,
                chain_name="A",
                mol_type=const.chain_type_ids["PROTEIN"],
                cluster_id=0,
                msa_id=-1,
                num_residues=10,
                entity_id=0,
            ),
            ChainInfo(
                chain_id=1,
                chain_name="L1",
                mol_type=const.chain_type_ids["NONPOLYMER"],
                cluster_id=1,
                msa_id=-1,
                num_residues=1,
                entity_id=1,
            ),
            ChainInfo(
                chain_id=2,
                chain_name="L2",
                mol_type=const.chain_type_ids["NONPOLYMER"],
                cluster_id=1,
                msa_id=-1,
                num_residues=1,
                entity_id=1,
            ),
        ],
        interfaces=[],
        affinity=AffinityInfo(chain_id=1, mw=123.4, chain_name="L1"),
    )


def test_expand_affinity_records_fans_out_repeated_binders():
    manifest = Manifest(records=[_make_affinity_record()])

    expanded = expand_affinity_records(manifest)

    assert len(expanded.records) == 2
    assert [record.affinity.chain_id for record in expanded.records] == [1, 2]
    assert [record.affinity.chain_name for record in expanded.records] == ["L1", "L2"]
    assert [record.affinity.output_id for record in expanded.records] == [
        "target_L1",
        "target_L2",
    ]


def test_expand_affinity_records_sanitizes_output_id():
    record = replace(
        _make_affinity_record(),
        chains=[
            _make_affinity_record().chains[0],
            replace(_make_affinity_record().chains[1], chain_name="L/1"),
            replace(_make_affinity_record().chains[2], chain_name="L 2"),
        ],
        affinity=AffinityInfo(chain_id=1, mw=123.4, chain_name="L/1"),
    )

    expanded = expand_affinity_records(Manifest(records=[record]))

    assert [record.affinity.output_id for record in expanded.records] == [
        "target_L_1",
        "target_L_2",
    ]


def test_expand_affinity_records_supports_legacy_chain_id_only():
    record = replace(_make_affinity_record(), affinity=AffinityInfo(chain_id=1, mw=123.4))
    manifest = Manifest(records=[record])

    expanded = expand_affinity_records(manifest)

    assert len(expanded.records) == 2
    assert [record.affinity.chain_name for record in expanded.records] == ["L1", "L2"]


def test_expand_affinity_records_raises_on_binder_mismatch():
    record = replace(
        _make_affinity_record(),
        affinity=AffinityInfo(chain_id=99, mw=123.4, chain_name="MISSING"),
    )

    with pytest.raises(
        click.ClickException,
        match="could not find binder chain 'MISSING'",
    ):
        expand_affinity_records(Manifest(records=[record]))


def test_filter_inputs_affinity_tracks_per_copy_outputs(tmp_path: Path):
    pred_dir = tmp_path / "predictions" / "target"
    pred_dir.mkdir(parents=True)
    (pred_dir / "pre_affinity_target.npz").write_bytes(b"ok")
    (pred_dir / "affinity_target_L1.json").write_text("{}")

    manifest = expand_affinity_records(Manifest(records=[_make_affinity_record()]))

    filtered = filter_inputs_affinity(manifest, tmp_path, override=False)

    assert len(filtered.records) == 1
    assert filtered.records[0].affinity.chain_name == "L2"


def test_filter_inputs_affinity_keeps_existing_outputs_with_override(tmp_path: Path):
    pred_dir = tmp_path / "predictions" / "target"
    pred_dir.mkdir(parents=True)
    (pred_dir / "pre_affinity_target.npz").write_bytes(b"ok")
    (pred_dir / "affinity_target_L1.json").write_text("{}")
    (pred_dir / "affinity_target_L2.json").write_text("{}")

    manifest = expand_affinity_records(Manifest(records=[_make_affinity_record()]))

    filtered = filter_inputs_affinity(manifest, tmp_path, override=True)

    assert len(filtered.records) == 2
