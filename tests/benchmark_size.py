"""Size limit benchmark for boltz-community.

Tests maximum protein size that can be processed on the current GPU
without running out of memory. Uses structure-only predictions (no affinity)
to isolate the size scaling behavior.

Usage:
    python tests/benchmark_size.py [--out results.json]
"""

import argparse
import json
import os
import subprocess
import tempfile
import time


# --- Test inputs at increasing sizes ---
# All use msa: empty and structure-only (no affinity) to focus on size scaling.
# Sequences are poly-alanine to avoid CCD lookup overhead.

def make_yaml(n_residues, ligand=True):
    """Generate a YAML input with a protein of n_residues and optional ligand."""
    seq = "A" * n_residues
    yaml = f"""\
version: 1
sequences:
  - protein:
      id: A
      sequence: {seq}
      msa: empty
"""
    if ligand:
        yaml += """\
  - ligand:
      id: L1
      smiles: 'NC(=N)c1ccccc1'
properties:
  - affinity:
      binder: L1
"""
    return yaml


CASES = [
    ("500res_struct", 500, False),
    ("500res_affinity", 500, True),
    ("1000res_struct", 1000, False),
    ("1000res_affinity", 1000, True),
    ("2000res_struct", 2000, False),
]


def run_benchmark(name, yaml_content, tmpdir, recycling_steps=3, diffusion_samples=1):
    """Run a single benchmark case and return timing/memory results."""
    input_path = os.path.join(tmpdir, f"{name}.yaml")
    with open(input_path, "w") as f:
        f.write(yaml_content)

    output_dir = os.path.join(tmpdir, f"output_{name}")
    cmd = [
        "boltz", "predict", input_path,
        "--out_dir", output_dir,
        "--recycling_steps", str(recycling_steps),
        "--diffusion_samples", str(diffusion_samples),
    ]

    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    elapsed = time.perf_counter() - start

    success = result.returncode == 0
    error = ""
    if not success:
        # Check for OOM
        combined = result.stdout + result.stderr
        if "out of memory" in combined.lower() or "CUDA out of memory" in combined:
            error = "OOM"
        else:
            error = result.stderr[-500:] if result.stderr else "unknown error"

    return {
        "name": name,
        "success": success,
        "wall_time_s": round(elapsed, 2),
        "returncode": result.returncode,
        "error": error,
    }


def get_gpu_info():
    """Get GPU info via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Boltz-community size benchmark")
    parser.add_argument("--out", default=None, help="Output JSON file path")
    args = parser.parse_args()

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info}")

    try:
        from importlib.metadata import version
        pkg_version = version("boltz-community")
    except Exception:
        pkg_version = "unknown"
    print(f"boltz-community version: {pkg_version}\n")

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        print("=" * 60)
        print("Size limit benchmark")
        print("=" * 60)
        for name, n_res, ligand in CASES:
            print(f"\nRunning: {name} ({n_res} residues) ...")
            yaml = make_yaml(n_res, ligand=ligand)
            r = run_benchmark(name, yaml, tmpdir)
            r["n_residues"] = n_res
            r["has_ligand"] = ligand
            results.append(r)
            status = "OK" if r["success"] else f"FAIL ({r['error'][:50]})"
            print(f"  {status} — {r['wall_time_s']}s")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    summary = {
        "gpu": gpu_info,
        "version": pkg_version,
        "results": results,
    }

    for r in results:
        status = "OK" if r["success"] else f"FAIL ({r['error'][:50]})"
        label = "struct+aff" if r["has_ligand"] else "struct"
        print(f"  {r['n_residues']:>5} res ({label:>10}): {r['wall_time_s']:>8.1f}s [{status}]")

    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
