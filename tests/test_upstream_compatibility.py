"""The compatibility boundary must fail closed before torch/model imports."""

import subprocess
import sys
from pathlib import Path

import pytest

from boltz.upstream import contract as source_contract


def test_reviewed_fork_sources_match_contract():
    c = source_contract.contract()
    root = Path(source_contract.__file__).resolve().parents[1]
    source_contract.verify_files(root, c["fork_files"])
    assert {p.relative_to(root).as_posix() for p in root.rglob("*.py")} == set(
        c["fork_files"]
    )


def test_modified_source_is_refused(tmp_path):
    p = tmp_path / "provider.py"
    p.write_text("old")
    expected = {p.name: source_contract.digest(p)}
    p.write_text("new")
    with pytest.raises(RuntimeError, match="source mismatch"):
        source_contract.verify_files(tmp_path, expected)


def test_missing_source_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="source mismatch"):
        source_contract.verify_files(tmp_path, {"missing.py": "0" * 64})


def test_reference_core_inventory():
    c = source_contract.contract()
    assert len(c["stock_files"]) == 18
    assert "boltz.model.layers.pairformer" in c["numeric_modules"]
    for name in c["numeric_modules"]:
        assert name.replace(".", "/") + ".py" in c["stock_files"]
    # Original forward is loaded separately; it must be in the same wheel.
    assert "boltz/model/models/boltz2.py" in c["stock_files"]


def test_bootstrap_import_does_not_import_torch():
    program = (
        "import sys; import boltz.upstream.bootstrap; assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", program], check=True)


def test_late_install_refused(monkeypatch, tmp_path):
    from boltz.upstream.bootstrap import install

    monkeypatch.setitem(sys.modules, "boltz.model.example", object())
    with pytest.raises(RuntimeError, match="fresh interpreter"):
        install(tmp_path)


def test_overlay_precedes_normal_path_loader(tmp_path):
    # Test in a fresh process, without importing any real model or requiring torch.
    program = """
import importlib.util, pathlib, sys
from boltz.upstream.bootstrap import install
root = pathlib.Path(sys.argv[1]); install(root)
spec = importlib.util.find_spec("boltz.main")
assert spec.origin == str(root / "boltz/main.py"), spec.origin
assert "torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", program, str(tmp_path)], check=True)


def test_foreign_already_loaded_engine_refused(monkeypatch, tmp_path):
    import types

    foreign = types.ModuleType("boltz2_opt")
    foreign.__file__ = str(tmp_path / "another-checkout/boltz2_opt/__init__.py")
    monkeypatch.setitem(sys.modules, "boltz2_opt", foreign)
    with pytest.raises(RuntimeError, match="another source"):
        source_contract.verify_loaded_engine(tmp_path / "pinned")


def test_late_parser_install_refused(monkeypatch, tmp_path):
    from boltz.upstream.bootstrap import install

    monkeypatch.setitem(sys.modules, "boltz.data.parse.csv", object())
    with pytest.raises(RuntimeError, match="fresh interpreter"):
        install(tmp_path)


def test_waste_proof_checks_actual_loaded_source(monkeypatch, tmp_path):
    import importlib.util
    import types

    from boltz.upstream.bootstrap import _Loader

    stock = tmp_path / "reference"
    relative = "boltz/model/layers/transition.py"
    original = stock / relative
    original.parent.mkdir(parents=True)
    original.write_text("# reviewed original body\n")
    stub = tmp_path / "boltz_waste_levers.py"
    stub.write_text(
        f"PINNED_SHA256 = {{{relative!r}: {source_contract.digest(original)!r}}}\n"
    )
    loader = _Loader("boltz_waste_levers", str(stub), stock)
    spec = importlib.util.spec_from_loader("boltz_waste_levers", loader)
    provider = importlib.util.module_from_spec(spec)
    loader.exec_module(provider)
    layer = types.ModuleType("boltz.model.layers.transition")
    layer.__file__ = str(original)
    monkeypatch.setitem(sys.modules, layer.__name__, layer)
    provider._check_pin()
    original.write_text("# changed numerical body\n")
    with pytest.raises(RuntimeError, match="source mismatch"):
        provider._check_pin()


def test_affinity_child_commands_keep_reference_bootstrap(tmp_path):
    import importlib.util

    from boltz.upstream.bootstrap import _Loader

    stub = tmp_path / "affinity_leg.py"
    stub.write_text(
        'def command(path, python="python"):\n'
        '    return [python, "-s", "-m", "boltz2_opt.affinity_leg", path]\n'
        'def server_command(python="python"):\n'
        '    return [python, "-s", "-m", "boltz2_opt.affinity_leg", "--serve"]\n'
    )
    loader = _Loader("boltz2_opt.affinity_leg", str(stub), tmp_path)
    spec = importlib.util.spec_from_loader("boltz2_opt.affinity_leg", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    prefix = [
        "custom-python",
        "-s",
        "-m",
        "boltz.upstream.worker_launch",
        "--reference-affinity",
    ]
    assert module.command("request.json", python="custom-python") == [
        *prefix,
        "request.json",
    ]
    assert module.server_command(python="custom-python") == [*prefix, "--serve"]


def test_internal_affinity_entrypoint_forwards_arguments(monkeypatch):
    import types

    from boltz.upstream import runtime, worker_launch

    calls = []
    monkeypatch.setattr(runtime, "configure", lambda **kw: calls.append(kw))
    package = types.ModuleType("boltz2_opt")
    package.__path__ = []
    affinity = types.ModuleType("boltz2_opt.affinity_leg")
    affinity.main = lambda argv: calls.append(argv) or 3
    monkeypatch.setitem(sys.modules, "boltz2_opt", package)
    monkeypatch.setitem(sys.modules, "boltz2_opt.affinity_leg", affinity)
    assert worker_launch.main(["--reference-affinity", "--serve"]) == 3
    assert calls == [{"overlay": True}, ["--serve"]]


def test_engine_and_numerical_sources_are_installed_together(monkeypatch):
    monkeypatch.setenv("BOLTZ_ANTHROPIC_ROOT", "/missing/external-checkout")
    root, numerical, detail = source_contract.prepare()
    assert root.is_relative_to(Path(source_contract.__file__).parents[1])
    assert numerical.is_relative_to(root.parent)
    assert detail["integrated_engine"] is True
    assert not list(root.rglob("*.whl"))
    for relative, wanted in source_contract.contract()["stock_files"].items():
        assert source_contract.digest(numerical / relative) == wanted


def test_original_profile_parser_and_numerics_have_not_changed():
    c = source_contract.contract()
    package = Path(source_contract.__file__).resolve().parents[2] / "boltz2_opt"
    for name in ("cli.py", "modes.py", "settings.py", "report.py", "worker.py"):
        assert (
            source_contract.digest(package / name)
            == c["upstream_engine_files"]["opt/boltz2_opt/" + name]
        )


def test_plain_import_leaves_optimizations_off():
    program = """
import sys
import boltz
import boltz2_opt
assert 'torch' not in sys.modules
assert 'opt_core' not in sys.modules
assert boltz2_opt.status()['active'] is False
assert not any(type(x).__name__ == '_CoreFinder' for x in sys.meta_path)
"""
    subprocess.run([sys.executable, "-c", program], check=True)


def test_native_engine_cli_reaches_integrated_entry(monkeypatch):
    from boltz.upstream import __main__ as integrated
    from boltz2_opt import __main__ as native

    called = []
    monkeypatch.setattr(integrated, "main", lambda argv: called.append(argv) or 3)
    assert native.main(["pred", "--mode", "exact"]) == 3
    assert called == [["pred", "--mode", "exact"]]


def test_prepare_uses_no_reference_archive(monkeypatch):
    import zipfile

    def no_archive(*args, **kwargs):
        raise AssertionError("Installed inference must not extract a reference wheel")

    monkeypatch.setattr(zipfile, "ZipFile", no_archive)
    assert source_contract.prepare()[2]["integrated_engine"] is True


def test_guidance_reference_is_scoped_to_engine_process():
    program = """
import importlib.util, pathlib, sys
root = pathlib.Path(sys.argv[1])
if sys.argv[2] == 'engine':
 from boltz.upstream.bootstrap import install
 install(root)
spec = importlib.util.find_spec('boltz.model.potentials.potentials')
reference = root / 'boltz/model/potentials/potentials.py'
assert (pathlib.Path(spec.origin) == reference) == (sys.argv[2] == 'engine'), spec.origin
assert 'torch' not in sys.modules
"""
    reference = source_contract.engine_root().parent / "reference"
    for mode in ("native", "engine"):
        subprocess.run(
            [sys.executable, "-c", program, str(reference), mode], check=True
        )


def test_reference_constraint_features_keep_interchain_connections():
    program = """
import numpy as np
from types import SimpleNamespace
from boltz.upstream.bootstrap import install
from boltz.upstream.contract import engine_root

install(engine_root().parent / "reference")
from boltz.data import const
from boltz.data.feature import featurizerv2
from boltz.data.types import BondV2, Chain

chains = np.zeros(2, dtype=Chain)
chains["entity_id"] = [0, 1]
covalent = const.bond_type_ids["COVALENT"]
bonds = np.array([
    (0, 0, 0, 1, 0, 1, covalent),
    (0, 1, 0, 2, 0, 2, covalent),
], dtype=BondV2)
data = SimpleNamespace(structure=SimpleNamespace(chains=chains, bonds=bonds))
result = featurizerv2.process_chain_feature_constraints(data, ref_element=None)
assert result["connected_atom_index"].tolist() == [[0], [2]], result
assert result["connected_chain_index"].tolist() == [[0], [1]], result
assert "connected_atom_lower_bounds" not in result, result
assert featurizerv2.process_residue_constraint_features.__module__ == (
    "boltz.upstream._reference_features"
)
"""
    subprocess.run([sys.executable, "-c", program], check=True)


def test_reference_affinity_filter_accepts_only_prepared_affinity_inputs(tmp_path):
    program = """
from pathlib import Path
import sys
from types import SimpleNamespace
import click
from boltz.upstream.bootstrap import install
from boltz.upstream.contract import engine_root

install(engine_root().parent / "reference")
from boltz.main import filter_inputs_affinity
from boltz.data.types import Manifest

output = Path(sys.argv[1])
structure = SimpleNamespace(id="structure", affinity=None)
affinity = SimpleNamespace(id="ligand", affinity=SimpleNamespace(output_id=None))
manifest = Manifest([structure, affinity])
try:
    filter_inputs_affinity(manifest, output)
except click.ClickException as error:
    assert "pre_affinity_" in str(error), error
else:
    raise AssertionError("missing affinity structure accepted")
folder = output / "predictions" / "ligand"
folder.mkdir(parents=True)
(folder / "pre_affinity_ligand.npz").touch()
assert filter_inputs_affinity(manifest, output).records == [affinity]
"""
    subprocess.run([sys.executable, "-c", program, str(tmp_path)], check=True)


def test_writer_overlap_retains_community_confidence_fields(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import torch

    from boltz.model.layers.structure_checks import CHECK_KEYS
    from boltz2_opt import writer

    monkeypatch.setattr(writer, "_Q", SimpleNamespace(put=lambda _: None))
    monkeypatch.setattr(writer, "_PENDING", {})
    monkeypatch.setattr(writer, "_INFLIGHT", [])
    monkeypatch.setattr(writer, "_SEQ", [0])
    monkeypatch.setattr(writer, "STATE", dict(writer.STATE))
    destination = SimpleNamespace(
        data_dir=tmp_path,
        output_dir=tmp_path,
        output_format="mmcif",
        boltz2=True,
        write_embeddings=False,
    )
    metrics = ("complex_pae", "complex_ipae", "chains_pae", "pair_chains_pae")
    prediction = {
        name: torch.arange(2)
        for name in (*metrics, *(f"{key}_unweighted" for key in metrics), *CHECK_KEYS)
    }
    job = writer.submit(
        destination, prediction, {"record": [SimpleNamespace(id="item")]}
    )
    for name, value in prediction.items():
        torch.testing.assert_close(job["host"][name], value)
