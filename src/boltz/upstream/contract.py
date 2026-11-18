"""Verify this distribution's integrated engine and reference numerical source."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

CONTRACT_PATH = Path(__file__).with_name("contract.json")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


def engine_root() -> Path:
    """Return the installed engine assets without importing torch or opt_core."""
    return Path(__file__).resolve().parents[1] / "_anthropic_engine" / "engine"


def verify_files(root: Path, expected: dict[str, str]) -> None:
    for relative, wanted in expected.items():
        path = root / relative
        if not path.is_file() or digest(path) != wanted:
            raise RuntimeError(f"integrated engine source mismatch: {path}")


def verify_loaded_engine(root: Path) -> None:
    """Refuse an engine imported from a separate installation."""
    for name, module in tuple(sys.modules.items()):
        if name == "boltz2_opt" or name.startswith("boltz2_opt."):
            filename = getattr(module, "__file__", None)
            if not filename or not Path(filename).resolve().is_relative_to(root):
                raise RuntimeError(f"engine already loaded from another source: {name}")


def prepare() -> tuple[Path, Path, dict]:
    """Verify integrated sources, then set transport paths for child processes."""
    c = contract()
    fork = Path(__file__).resolve().parents[1]
    root = engine_root()
    stock = root.parent / "reference"
    package = fork.parent / "boltz2_opt"
    verify_loaded_engine(package)
    verify_files(root, c["engine_files"])
    verify_files(package, c["engine_package_files"])
    verify_files(stock, c["stock_files"])
    verify_files(fork, c["fork_files"])
    actual = {p.relative_to(fork).as_posix() for p in fork.rglob("*.py")}
    if actual != set(c["fork_files"]):
        raise RuntimeError(
            "fork source inventory differs from the reviewed engine contract"
        )
    paths = [str(fork.parent), *os.environ.get("PYTHONPATH", "").split(os.pathsep)]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(p for p in paths if p))
    os.environ["MODEL_OPT"] = str(root)
    detail = {
        "version": c["fork_version"],
        "root": str(fork),
        "pinned": False,
        "compatible_fork": True,
        "integrated_engine": True,
        "reference_commit": c["upstream_commit"],
        "reference_wheel_sha256": c["stock_wheel_sha256"],
        "fork_source_sha256": hashlib.sha256(
            json.dumps(c["fork_files"], sort_keys=True).encode()
        ).hexdigest(),
        "numeric_overlays": c["numeric_modules"],
        "function_overlays": [
            "boltz.model.models.boltz2.Boltz2.forward",
            "boltz.data.feature.featurizerv2.construct_paired_msa",
            "boltz.data.feature.featurizerv2.process_residue_constraint_features",
            "boltz.data.feature.featurizerv2.process_chain_feature_constraints",
            "boltz.main.filter_inputs_affinity (fork completeness check)",
            "boltz_waste_levers._check_pin (packaged numerical module paths)",
        ],
    }
    return root, stock, detail
