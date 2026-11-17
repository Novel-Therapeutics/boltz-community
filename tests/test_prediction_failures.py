"""Prediction failures must be visible to CLI callers after successful writes."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pytorch_lightning as pl
import torch
from click.testing import CliRunner
from pytorch_lightning import LightningModule

import boltz.main as main
from boltz.data import const
from boltz.data.module import inference, inferencev2
from boltz.data.types import AffinityInfo, ChainInfo, Manifest, Record, StructureInfo
from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter, PredictionError
from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2


@pytest.fixture(autouse=True)
def _restore_torch_settings():
    enabled = torch.is_grad_enabled()
    precision = torch.get_float32_matmul_precision()
    load = torch.load
    yield
    torch.set_grad_enabled(enabled)
    torch.set_float32_matmul_precision(precision)
    torch.load = load


def _writer(tmp_path, affinity=False):
    if affinity:
        return BoltzAffinityWriter(tmp_path, tmp_path / "out")
    return BoltzWriter(tmp_path, tmp_path / "out", boltz2=True)


def _trainer(remote_failed=0):
    reduce = Mock(side_effect=lambda value, **_: value + remote_failed)
    return SimpleNamespace(strategy=SimpleNamespace(reduce=reduce))


@pytest.mark.parametrize("affinity", [False, True])
def test_failed_batch_counts_each_example_and_epoch_fails(tmp_path, affinity):
    writer = _writer(tmp_path, affinity)
    batch = {"record": [SimpleNamespace(id="one"), SimpleNamespace(id="two")]}
    writer.write_on_batch_end(None, None, {"exception": True}, [], batch, 0, 0)
    assert writer.failed == 2
    with pytest.raises(PredictionError, match="2.*prediction.*failed"):
        writer.on_predict_epoch_end(_trainer(), SimpleNamespace(device="cpu"))


@pytest.mark.parametrize("affinity", [False, True])
def test_remote_failure_is_not_hidden_by_a_successful_local_rank(tmp_path, affinity):
    writer = _writer(tmp_path, affinity)
    trainer = _trainer(remote_failed=1)
    with pytest.raises(PredictionError, match="1.*prediction.*failed"):
        writer.on_predict_epoch_end(trainer, SimpleNamespace(device="cpu"))
    value = trainer.strategy.reduce.call_args.args[0]
    assert value.device.type == "cpu"
    assert value.dtype == torch.int64
    assert trainer.strategy.reduce.call_args.kwargs == {"reduce_op": "sum"}


@pytest.mark.parametrize("affinity", [False, True])
def test_clean_epoch_still_participates_in_global_failure_check(tmp_path, affinity):
    writer = _writer(tmp_path, affinity)
    trainer = _trainer()
    writer.on_predict_epoch_end(trainer, SimpleNamespace(device="cpu"))
    trainer.strategy.reduce.assert_called_once()


@pytest.mark.parametrize("affinity", [False, True])
def test_global_check_keeps_local_count_and_reuse_starts_clean(tmp_path, affinity):
    writer = _writer(tmp_path, affinity)
    writer.write_on_batch_end(
        None, None, {"exception": True}, [], {"record": [None, None]}, 0, 0
    )
    trainer = _trainer(remote_failed=1)
    module = SimpleNamespace(device="cpu")
    for _ in range(2):
        with pytest.raises(PredictionError, match="3 prediction"):
            writer.on_predict_epoch_end(trainer, module)
    assert writer.failed == 2
    writer.on_predict_start(trainer, module)
    writer.on_predict_epoch_end(_trainer(), module)
    assert writer.failed == 0


@pytest.mark.parametrize("affinity", [False, True])
def test_failed_write_keeps_existing_outputs(tmp_path, affinity):
    writer = _writer(tmp_path, affinity)
    dest = writer.output_dir / "saved"
    dest.mkdir()
    artifact = dest / ("affinity_saved.json" if affinity else "saved_model_0.cif")
    artifact.write_bytes(b"previous valid result")
    writer.write_on_batch_end(
        None,
        None,
        {"exception": True},
        [],
        {"record": [SimpleNamespace(id="saved")]},
        0,
        0,
    )
    with pytest.raises(PredictionError):
        writer.on_predict_epoch_end(_trainer(), SimpleNamespace(device="cpu"))
    assert artifact.read_bytes() == b"previous valid result"


def _run_cli(
    monkeypatch,
    tmp_path,
    structure_outcomes,
    affinity_outcomes=(),
    model="boltz2",
    *,
    existing=None,
    missing_pre=(),
    devices="1",
    trainer_configs=None,
    batch_size=1,
):
    (tmp_path / "input.yaml").write_text("version: 1\n")
    records = [
        Record(
            id=f"case{i}",
            structure=StructureInfo(),
            interfaces=[],
            chains=[
                ChainInfo(
                    chain_id=0,
                    chain_name="L",
                    mol_type=const.chain_type_ids["NONPOLYMER"],
                    cluster_id=0,
                    msa_id=-1,
                    num_residues=1,
                    entity_id=0,
                )
            ],
            affinity=AffinityInfo(chain_id=0, mw=100.0) if affinity_outcomes else None,
        )
        for i in range(len(structure_outcomes))
    ]
    manifest = Manifest(records=records)
    calls = []
    output = tmp_path / "boltz_results_input" / "predictions"
    for idx, files in (existing or {}).items():
        dest = output / records[idx].id
        dest.mkdir(parents=True, exist_ok=True)
        for name in files:
            (dest / name).write_text("existing output\n")
    monkeypatch.setattr(main, "download_boltz1", lambda *_: None)
    monkeypatch.setattr(main, "download_boltz2", lambda *_: None)
    monkeypatch.setattr(main, "check_inputs", lambda data, **_: [data])
    monkeypatch.setattr(main, "process_inputs", lambda **_: None)
    monkeypatch.setattr(Manifest, "load", lambda *_: manifest)
    monkeypatch.setattr(inference, "BoltzInferenceDataModule", lambda **kw: kw)
    monkeypatch.setattr(inferencev2, "Boltz2InferenceDataModule", lambda **kw: kw)
    monkeypatch.setattr(main, "_validate_checkpoint_for_load", lambda *_: None)
    monkeypatch.setattr(
        Boltz1,
        "load_from_checkpoint",
        lambda *a, **kw: SimpleNamespace(eval=lambda: None),
    )
    monkeypatch.setattr(
        Boltz2,
        "load_from_checkpoint",
        lambda *a, **kw: SimpleNamespace(eval=lambda: None),
    )

    class Trainer:
        def __init__(self, callbacks, **kwargs):
            self.callbacks = callbacks
            self.strategy = _trainer().strategy
            if trainer_configs is not None:
                trainer_configs.append(kwargs)

        def predict(self, *_, datamodule, **__):
            writer = self.callbacks[0]
            writer.on_predict_start(self, SimpleNamespace(device="cpu"))
            affinity = isinstance(writer, BoltzAffinityWriter)
            calls.append("affinity" if affinity else "structure")
            outcomes = affinity_outcomes if affinity else structure_outcomes
            for record in datamodule["manifest"].records:
                idx = int(record.id.removeprefix("case"))
                failed = outcomes[idx]
                batch = {"record": [record]}
                if failed:
                    writer.write_on_batch_end(
                        None, None, {"exception": True}, [], batch, idx, 0
                    )
                elif affinity:
                    writer.write_on_batch_end(
                        None,
                        None,
                        {
                            "exception": False,
                            "affinity_pred_value": torch.tensor(1.25),
                            "affinity_probability_binary": torch.tensor(0.75),
                        },
                        [],
                        batch,
                        idx,
                        0,
                    )
                else:
                    dest = writer.output_dir / record.id
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest / "saved.cif").write_text("successful prediction\n")
                    if idx not in missing_pre and record.affinity:
                        (dest / f"pre_affinity_{record.id}.npz").write_bytes(b"input")
            writer.on_predict_epoch_end(self, SimpleNamespace(device="cpu"))

    monkeypatch.setattr(pl, "Trainer", Trainer)
    result = CliRunner().invoke(
        main.cli,
        [
            "predict",
            str(tmp_path / "input.yaml"),
            "--model",
            model,
            "--cache",
            str(tmp_path / "cache"),
            "--out_dir",
            str(tmp_path),
            "--accelerator",
            "cpu",
            "--devices",
            devices,
            "--batch_size",
            str(batch_size),
            "--num_workers",
            "0",
        ],
    )
    return result, calls, output


@pytest.mark.parametrize("model", ["boltz1", "boltz2"])
@pytest.mark.parametrize("outcomes", [[True], [False, True], [True, False]])
def test_cli_structure_failure_is_nonzero_without_traceback(
    monkeypatch, tmp_path, model, outcomes
):
    result, calls, output = _run_cli(monkeypatch, tmp_path, outcomes, model=model)
    assert result.exit_code == 1, result.output
    assert "prediction" in result.output and "failed" in result.output
    assert isinstance(result.exception, SystemExit)
    assert calls == ["structure"]
    for idx, failed in enumerate(outcomes):
        if not failed:
            assert (
                output / f"case{idx}" / "saved.cif"
            ).read_text() == "successful prediction\n"


def test_structure_failure_stops_affinity_before_missing_input_is_read(
    monkeypatch, tmp_path
):
    result, calls, _ = _run_cli(monkeypatch, tmp_path, [False, True], [False, False])
    assert result.exit_code == 1
    assert calls == ["structure"]
    assert "FileNotFound" not in result.output


def test_affinity_failure_keeps_successful_files_and_returns_nonzero(
    monkeypatch, tmp_path
):
    result, calls, output = _run_cli(
        monkeypatch, tmp_path, [False, False], [False, True]
    )
    assert result.exit_code == 1, result.output
    assert calls == ["structure", "affinity"]
    saved = json.loads((output / "case0" / "affinity_case0.json").read_text())
    assert saved["affinity_pred_value"] == 1.25
    assert (output / "case1" / "saved.cif").exists()
    assert not (output / "case1" / "affinity_case1.json").exists()


@pytest.mark.parametrize("affinity", [False, True])
def test_successful_cli_keeps_exit_zero(monkeypatch, tmp_path, affinity):
    result, calls, output = _run_cli(
        monkeypatch, tmp_path, [False], [False] if affinity else []
    )
    assert result.exit_code == 0, result.output
    assert calls == (["structure", "affinity"] if affinity else ["structure"])
    assert (output / "case0" / "saved.cif").exists()


def test_cli_skips_completed_structure_and_affinity(monkeypatch, tmp_path):
    configs = []
    result, calls, output = _run_cli(
        monkeypatch,
        tmp_path,
        [True],
        [True],
        existing={0: ("saved.cif", "pre_affinity_case0.npz", "affinity_case0.json")},
        trainer_configs=configs,
    )
    assert result.exit_code == 0, result.output
    assert calls == []
    assert configs == []
    for path in (output / "case0").iterdir():
        assert path.read_text() == "existing output\n"


@pytest.mark.parametrize("failed", [False, True])
def test_cli_affinity_only_after_skipped_structure(monkeypatch, tmp_path, failed):
    result, calls, output = _run_cli(
        monkeypatch,
        tmp_path,
        [True],
        [failed],
        existing={0: ("saved.cif", "pre_affinity_case0.npz")},
    )
    assert result.exit_code == int(failed), result.output
    assert calls == ["affinity"]
    assert (output / "case0" / "saved.cif").read_text() == "existing output\n"
    assert (output / "case0" / "affinity_case0.json").exists() is not failed


@pytest.mark.parametrize("missing", [(0,), (0, 1)])
def test_missing_affinity_input_cannot_turn_into_success(
    monkeypatch, tmp_path, missing
):
    result, calls, output = _run_cli(
        monkeypatch,
        tmp_path,
        [False, False],
        [False, False],
        missing_pre=missing,
    )
    assert result.exit_code == 1, result.output
    assert "pre_affinity_*.npz missing" in result.output
    assert calls == ["structure"]
    assert (output / "case0" / "saved.cif").exists()
    assert (output / "case1" / "saved.cif").exists()


@pytest.mark.parametrize(
    "devices, expected", [("4", [3, 2]), ("1,3,5,7", [[1, 3, 5], [1, 3]])]
)
def test_cli_sizes_each_phase_after_filtering(monkeypatch, tmp_path, devices, expected):
    configs = []
    result, calls, output = _run_cli(
        monkeypatch,
        tmp_path,
        [False, False, False],
        [True, False, False],
        existing={0: ("affinity_case0.json",)},
        devices=devices,
        trainer_configs=configs,
        batch_size=2,
    )
    assert result.exit_code == 0, result.output
    assert calls == ["structure", "affinity"]
    assert [config["devices"] for config in configs] == expected
    assert (output / "case0" / "affinity_case0.json").read_text() == "existing output\n"


def test_cli_sizes_affinity_when_all_structure_is_existing(monkeypatch, tmp_path):
    configs = []
    result, calls, _ = _run_cli(
        monkeypatch,
        tmp_path,
        [True, True],
        [True, False],
        devices="4",
        existing={
            0: ("pre_affinity_case0.npz", "affinity_case0.json"),
            1: ("pre_affinity_case1.npz",),
        },
        trainer_configs=configs,
    )
    assert result.exit_code == 0, result.output
    assert calls == ["affinity"]
    assert [config["devices"] for config in configs] == [1]


@pytest.mark.parametrize("model_class", [Boltz1, Boltz2])
def test_actual_model_oom_path_reaches_failure_signal(monkeypatch, model_class):
    model = model_class.__new__(model_class)
    LightningModule.__init__(model)
    model.predict_args = dict(
        recycling_steps=1, sampling_steps=1, diffusion_samples=1, max_parallel_samples=1
    )

    def fail(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory: controlled test")

    monkeypatch.setattr(model, "forward", fail)
    assert model.predict_step({}, 0) == {"exception": True}


@pytest.mark.parametrize("model_class", [Boltz1, Boltz2])
def test_unrelated_model_error_is_not_swallowed(monkeypatch, model_class):
    model = model_class.__new__(model_class)
    LightningModule.__init__(model)
    model.predict_args = dict(
        recycling_steps=1, sampling_steps=1, diffusion_samples=1, max_parallel_samples=1
    )

    def fail(*args, **kwargs):
        raise RuntimeError("unrelated model error")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="unrelated model error"):
        model.predict_step({}, 0)


@pytest.mark.parametrize("affinity", [False, True])
@pytest.mark.parametrize("failure", ["none", "remote", "both"])
@pytest.mark.skipif(
    __import__("sys").platform == "win32", reason="local Gloo/fork probe"
)
def test_real_two_rank_failure_reduction(
    tmp_path, minimal_structure, affinity, failure
):
    """Count uneven shards and batched failures exactly once on every rank."""
    import os
    import subprocess
    import sys
    import textwrap

    records = [
        Record(
            id=str(i),
            structure=StructureInfo(),
            interfaces=[],
            chains=[
                ChainInfo(
                    chain_id=0,
                    chain_name="A",
                    mol_type=const.chain_type_ids["PROTEIN"],
                    cluster_id=0,
                    msa_id=-1,
                    num_residues=3,
                )
            ],
        )
        for i in range(5)
    ]
    Manifest(records).dump(tmp_path / "manifest.json")
    for record in records:
        minimal_structure.dump(tmp_path / f"{record.id}.npz")

    script = textwrap.dedent("""
        import sys
        from pathlib import Path
        import torch
        from torch.utils.data import DataLoader
        from pytorch_lightning import LightningModule, Trainer
        from pytorch_lightning.strategies import DDPStrategy
        from boltz.data.types import Manifest, Structure
        from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter, PredictionError

        root = Path(sys.argv[1])
        failure = sys.argv[2]
        affinity = sys.argv[3] == "True"

        class FailureReceipt:
            def on_predict_epoch_end(self, trainer, module):
                try:
                    super().on_predict_epoch_end(trainer, module)
                except PredictionError as error:
                    (root / f"rank_{trainer.global_rank}.txt").write_text(str(error))
                    trainer.strategy.barrier()
                    raise

        class StructureWriter(FailureReceipt, BoltzWriter):
            pass

        class AffinityWriter(FailureReceipt, BoltzAffinityWriter):
            pass

        class Model(LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
            def transfer_batch_to_device(self, batch, device, dataloader_idx):
                # Metadata records are frozen dataclasses, as in the real data module.
                return batch
            def predict_step(self, batch, batch_idx, dataloader_idx=0):
                if failure == "both" or (failure == "remote" and self.global_rank == 1):
                    return {"exception": True}
                if not affinity:
                    coords = torch.tensor(Structure.load(root / "0.npz").atoms["coords"])
                    size = len(batch["record"])
                    return {"exception": False, "coords": coords.unsqueeze(0).repeat(size, 1, 1),
                            "masks": torch.ones(size, coords.shape[0])}
                return {"exception": False, "affinity_pred_value": torch.tensor(1.25),
                        "affinity_probability_binary": torch.tensor(.75)}

        writer = (AffinityWriter if affinity else StructureWriter)(root, root / "out")
        trainer = Trainer(accelerator="cpu", devices=2,
            strategy=DDPStrategy(start_method="fork", process_group_backend="gloo"), callbacks=[writer],
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False)
        loader = DataLoader(Manifest.load(root / "manifest.json").records,
                            batch_size=1 if affinity else 2,
                            collate_fn=lambda records: {"record": records})
        trainer.predict(Model(), dataloaders=loader, return_predictions=False)
    """)
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), failure, str(affinity)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    if failure == "none":
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, result.stdout + result.stderr
        expected = 2 if failure == "remote" else 5
        for rank in range(2):
            assert (
                f"{expected} prediction(s) failed"
                in (tmp_path / f"rank_{rank}.txt").read_text()
            )
    for idx in range(5):
        path = (
            tmp_path
            / "out"
            / str(idx)
            / (f"affinity_{idx}.json" if affinity else f"{idx}_model_0.cif")
        )
        failed = failure == "both" or (failure == "remote" and idx % 2 == 1)
        assert path.exists() is not failed
        if not failed:
            if affinity:
                assert json.loads(path.read_text())["affinity_pred_value"] == 1.25
            else:
                assert "_atom_site." in path.read_text()
