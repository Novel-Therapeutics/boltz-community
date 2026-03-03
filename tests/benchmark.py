"""Performance benchmark for boltz-community.

Runs boltz predict on representative co-folding inputs and reports
wall-clock time and peak GPU memory for each case.

Usage:
    python tests/benchmark.py [--out results.json]
"""

import argparse
import json
import os
import subprocess
import tempfile
import time


# --- Test inputs -----------------------------------------------------------
# Each case is (name, yaml_content, extra_args).
# All use boltz2 (default) with protein-ligand co-folding + affinity,
# which is the primary use case.

# Small: 10-residue peptide + tyrosine ligand with affinity
SMALL_PROTEIN_LIGAND = """\
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

# Medium: ~150-residue T4 lysozyme + benzamidine ligand with affinity
# T4 lysozyme (PDB 3HTB) is a classic protein-ligand benchmark target
MEDIUM_PROTEIN_LIGAND = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL
      msa: empty
  - ligand:
      id: L1
      smiles: 'NC(=N)c1ccccc1'
properties:
  - affinity:
      binder: L1
"""

# Large: ~300-residue carbonic anhydrase II + sulfonamide inhibitor with affinity
# CAII (PDB 1CA2) is a common drug target benchmark
LARGE_PROTEIN_LIGAND = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK
      msa: empty
  - ligand:
      id: L1
      smiles: 'NS(=O)(=O)c1ccc(N)cc1'
properties:
  - affinity:
      binder: L1
"""

CASES = [
    ("small_protein_ligand", SMALL_PROTEIN_LIGAND),
    ("medium_protein_ligand", MEDIUM_PROTEIN_LIGAND),
    ("large_protein_ligand", LARGE_PROTEIN_LIGAND),
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
        timeout=1200,
    )
    elapsed = time.perf_counter() - start

    success = result.returncode == 0
    if not success:
        print(f"  FAILED: {result.stderr[-500:]}")

    return {
        "name": name,
        "success": success,
        "wall_time_s": round(elapsed, 2),
        "returncode": result.returncode,
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
    parser = argparse.ArgumentParser(description="Boltz-community benchmark")
    parser.add_argument("--out", default=None, help="Output JSON file path")
    args = parser.parse_args()

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info}")

    # Get package version
    try:
        from importlib.metadata import version
        pkg_version = version("boltz-community")
    except Exception:
        pkg_version = "unknown"
    print(f"boltz-community version: {pkg_version}\n")

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Phase 1: Individual runs (includes cold start for first) ---
        print("=" * 60)
        print("Phase 1: Individual predictions (cold start + single runs)")
        print("=" * 60)
        for i, (name, yaml) in enumerate(CASES):
            label = f"{name} (cold start)" if i == 0 else name
            print(f"\nRunning: {label} ...")
            r = run_benchmark(name, yaml, tmpdir)
            r["phase"] = "individual"
            r["is_cold_start"] = (i == 0)
            results.append(r)
            print(f"  {'OK' if r['success'] else 'FAIL'} — {r['wall_time_s']}s")

        # --- Phase 2: Batch throughput (5 sequential small+medium runs) ---
        print(f"\n{'=' * 60}")
        print("Phase 2: Batch throughput (5 sequential medium protein-ligand)")
        print("=" * 60)
        batch_times = []
        for i in range(5):
            print(f"\nBatch run {i+1}/5 ...")
            r = run_benchmark(
                f"batch_{i}", MEDIUM_PROTEIN_LIGAND, tmpdir,
            )
            r["phase"] = "batch"
            r["batch_index"] = i
            results.append(r)
            if r["success"]:
                batch_times.append(r["wall_time_s"])
            print(f"  {'OK' if r['success'] else 'FAIL'} — {r['wall_time_s']}s")

        if batch_times:
            avg = sum(batch_times) / len(batch_times)
            print(f"\nBatch average: {avg:.2f}s per prediction")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    summary = {
        "gpu": gpu_info,
        "version": pkg_version,
        "results": results,
    }

    # Add computed metrics
    individual = [r for r in results if r["phase"] == "individual"]
    if individual and individual[0]["success"]:
        summary["cold_start_s"] = individual[0]["wall_time_s"]
    batch_successful = [r["wall_time_s"] for r in results
                        if r["phase"] == "batch" and r["success"]]
    if batch_successful:
        summary["batch_avg_s"] = round(sum(batch_successful) / len(batch_successful), 2)

    for r in individual:
        status = "OK" if r["success"] else "FAIL"
        cold = " (cold start)" if r.get("is_cold_start") else ""
        print(f"  {r['name']}{cold}: {r['wall_time_s']}s [{status}]")
    if batch_successful:
        print(f"  batch avg (medium, n={len(batch_successful)}): {summary.get('batch_avg_s')}s")

    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
