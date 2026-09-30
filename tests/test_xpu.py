"""XPU (Intel GPU) tests.

The accelerator-registration tests at the top run anywhere, including CI.
The prediction tests require an Intel GPU with a PyTorch XPU build and are
excluded from CI; run them manually:

    pytest tests/test_xpu.py -m xpu -v
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

from boltz.xpu import XPUAccelerator, register_xpu_accelerator, xpu_available

# --- Accelerator plumbing (no GPU needed) ---------------------------------


def test_register_xpu_accelerator():
    """The xpu accelerator is registered with Lightning, and re-registering is a no-op."""
    from pytorch_lightning.accelerators import AcceleratorRegistry

    register_xpu_accelerator()
    register_xpu_accelerator()
    assert "xpu" in AcceleratorRegistry.available_accelerators()
    assert XPUAccelerator.name() == "xpu"


@pytest.mark.parametrize(
    ("devices", "expected"),
    [(1, [0]), (2, [0, 1]), ([1], [1]), ("0,2", [0, 2])],
)
def test_parse_devices(devices, expected):
    assert XPUAccelerator.parse_devices(devices) == expected


def test_get_parallel_devices():
    assert XPUAccelerator.get_parallel_devices([0, 1]) == [
        torch.device("xpu", 0),
        torch.device("xpu", 1),
    ]


# --- End-to-end prediction (Intel GPU required) ---------------------------


def _run_boltz_predict(input_yaml, input_filename, tmpdir, extra_args=None):
    """Run boltz predict with --accelerator xpu and return (result, predictions_dir)."""
    input_path = os.path.join(tmpdir, input_filename)
    with open(input_path, "w") as f:
        f.write(input_yaml)

    output_dir = os.path.join(tmpdir, "output")
    cmd = [
        sys.executable, "-c", "from boltz.main import cli; cli()", "predict", input_path,
        "--out_dir", output_dir,
        "--accelerator", "xpu",
        "--recycling_steps", "1",
        "--diffusion_samples", "1",
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    stem = os.path.splitext(input_filename)[0]
    pred_dir = os.path.join(output_dir, f"boltz_results_{stem}", "predictions")
    return result, pred_dir


def _find_files(pred_dir, extension):
    """Find all files with given extension in prediction subdirectories."""
    found = []
    if not os.path.isdir(pred_dir):
        return found
    for subdir in os.listdir(pred_dir):
        subdir_path = os.path.join(pred_dir, subdir)
        if os.path.isdir(subdir_path):
            for f in os.listdir(subdir_path):
                if f.endswith(extension):
                    found.append(os.path.join(subdir_path, f))
    return found


@pytest.mark.xpu
@pytest.mark.skipif(not xpu_available(), reason="XPU not available")
def test_autocast_device_type_xpu():
    """autocast_device_type keeps "xpu", so the fp32 guards actually disable XPU autocast."""
    from boltz.model.modules.utils import autocast_device_type

    assert autocast_device_type("xpu") == "xpu"
    x = torch.ones(8, 8, device="xpu")
    with torch.autocast("xpu", dtype=torch.bfloat16):
        with torch.autocast(autocast_device_type("xpu"), enabled=False):
            assert (x @ x).dtype == torch.float32


@pytest.mark.xpu
@pytest.mark.skipif(not xpu_available(), reason="XPU not available")
def test_predict_peptide_xpu():
    """Run boltz predict on XPU with a small peptide."""
    input_yaml = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: ACDEFGHIKL
      msa: empty
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, pred_dir = _run_boltz_predict(input_yaml, "test_xpu.yaml", tmpdir)

        assert result.returncode == 0, (
            f"boltz predict --accelerator xpu failed:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        cif_files = _find_files(pred_dir, ".cif")
        assert len(cif_files) > 0, "No .cif output files found"
        assert os.path.getsize(cif_files[0]) > 0, "CIF file is empty"
        assert len(_find_files(pred_dir, ".json")) > 0, "No confidence JSON files found"


@pytest.mark.xpu
@pytest.mark.skipif(not xpu_available(), reason="XPU not available")
def test_predict_affinity_xpu():
    """Run boltz predict on XPU with protein+ligand affinity."""
    input_yaml = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: ACDEFGHIKL
      msa: empty
  - ligand:
      id: L1
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
properties:
  - affinity:
      binder: L1
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, pred_dir = _run_boltz_predict(input_yaml, "test_xpu_affinity.yaml", tmpdir)

        assert result.returncode == 0, (
            f"boltz predict --accelerator xpu failed:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        affinity_file = next(
            (f for f in _find_files(pred_dir, ".json")
             if os.path.basename(f).startswith("affinity_")),
            None,
        )
        assert affinity_file is not None, "No affinity_*.json found"
        with open(affinity_file) as f:
            affinity_data = json.load(f)

        prob = affinity_data.get("affinity_probability_binary")
        assert isinstance(prob, (int, float)) and 0.0 <= prob <= 1.0
        pred_val = affinity_data.get("affinity_pred_value")
        assert isinstance(pred_val, (int, float)) and -10.0 <= pred_val <= 10.0


@pytest.mark.xpu
@pytest.mark.skipif(not xpu_available(), reason="XPU not available")
def test_seed_is_byte_reproducible_xpu():
    """With --seed, two XPU runs write byte-identical structures (as on CUDA)."""
    input_yaml = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: ACDEFGHIKL
      msa: empty
"""
    outputs = []
    for run in range(2):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, pred_dir = _run_boltz_predict(
                input_yaml, f"test_xpu_seed{run}.yaml", tmpdir, ["--seed", "42"]
            )
            assert result.returncode == 0, result.stderr
            cif_files = _find_files(pred_dir, ".cif")
            assert cif_files, "No .cif output files found"
            with open(cif_files[0]) as f:
                # the input name is part of the file; compare the atom records only
                outputs.append([l for l in f if l.startswith(("ATOM", "HETATM"))])
    assert outputs[0] == outputs[1], "XPU prediction is not reproducible with --seed"
