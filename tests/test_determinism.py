"""Determinism test: verify boltz predict produces identical output given the same seed.

Runs boltz predict twice on the same input with --seed 42 and asserts the
output CIF files are byte-identical.

Marked as slow since it downloads the model and runs inference twice.
Run with: pytest tests/test_determinism.py -m slow
"""

import os
import subprocess
import tempfile

import pytest


def _run_boltz_predict(input_yaml, input_filename, tmpdir, extra_args=None):
    """Run boltz predict and return (result, predictions_dir)."""
    input_path = os.path.join(tmpdir, input_filename)
    with open(input_path, "w") as f:
        f.write(input_yaml)

    output_dir = os.path.join(tmpdir, "output")
    cmd = [
        "boltz", "predict", input_path,
        "--out_dir", output_dir,
        "--recycling_steps", "1",
        "--diffusion_samples", "1",
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

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
    return sorted(found)


INPUT_YAML = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: ACDEFGHIKL
      msa: empty
"""


@pytest.mark.slow
def test_predict_determinism_with_seed():
    """Run boltz predict twice with --seed 42 and verify byte-identical CIF output."""
    cif_contents = []

    for run_idx in range(2):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, pred_dir = _run_boltz_predict(
                INPUT_YAML,
                "test_determinism.yaml",
                tmpdir,
                extra_args=["--seed", "42"],
            )

            assert result.returncode == 0, (
                f"boltz predict failed on run {run_idx + 1}:\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

            assert os.path.isdir(pred_dir), (
                f"Predictions directory not found on run {run_idx + 1}: {pred_dir}"
            )

            cif_files = _find_files(pred_dir, ".cif")
            assert len(cif_files) > 0, (
                f"No .cif output files found on run {run_idx + 1}"
            )

            # Read all CIF file contents (sorted by path for stable ordering)
            run_contents = []
            for cif_path in cif_files:
                with open(cif_path, "rb") as f:
                    run_contents.append((os.path.basename(cif_path), f.read()))
            cif_contents.append(run_contents)

    # Both runs should produce the same number of CIF files
    assert len(cif_contents[0]) == len(cif_contents[1]), (
        f"Different number of CIF files: run 1 produced {len(cif_contents[0])}, "
        f"run 2 produced {len(cif_contents[1])}"
    )

    # Compare each CIF file byte-for-byte
    for (name1, data1), (name2, data2) in zip(cif_contents[0], cif_contents[1]):
        assert name1 == name2, (
            f"CIF filenames differ: {name1!r} vs {name2!r}"
        )
        assert data1 == data2, (
            f"CIF file {name1!r} differs between runs "
            f"(run 1: {len(data1)} bytes, run 2: {len(data2)} bytes). "
            f"Prediction is not deterministic with --seed 42."
        )
