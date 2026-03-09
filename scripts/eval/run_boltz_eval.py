"""Run OpenStructure evaluation on Boltz predictions.

Evaluates Boltz-1 or Boltz-2 predictions against reference structures using
OpenStructure v2.8.0 Docker image. Works with any number of samples.

Usage:
    python scripts/eval/run_boltz_eval.py \
        predictions/ targets/ evals/ \
        --mount /data --num-samples 5
"""

import argparse
import concurrent.futures
import subprocess
from pathlib import Path

from tqdm import tqdm


OST_COMPARE_STRUCTURE = r"""
#!/bin/bash
IMAGE_NAME=openstructure-0.2.8

command="compare-structures \
-m {model_file} \
-r {reference_file} \
--fault-tolerant \
--min-pep-length 4 \
--min-nuc-length 4 \
-o {output_path} \
--lddt --bb-lddt --qs-score --dockq \
--ics --ips --rigid-scores --patch-scores --tm-score"

docker run -u $(id -u):$(id -g) --rm --volume {mount}:{mount} $IMAGE_NAME $command
"""

OST_COMPARE_LIGAND = r"""
#!/bin/bash
IMAGE_NAME=openstructure-0.2.8

command="compare-ligand-structures \
-m {model_file} \
-r {reference_file} \
--fault-tolerant \
--lddt-pli --rmsd \
--substructure-match \
-o {output_path}"

docker run -u $(id -u):$(id -g) --rm --volume {mount}:{mount} $IMAGE_NAME $command
"""


def evaluate_structure(
    name: str,
    pred: str,
    reference: str,
    outdir: str,
    mount: str,
    executable: str = "/bin/bash",
) -> None:
    """Evaluate a single prediction against reference."""
    # Polymer metrics
    out_path = Path(outdir) / f"{name}.json"
    if not out_path.exists():
        subprocess.run(
            OST_COMPARE_STRUCTURE.format(
                model_file=pred,
                reference_file=reference,
                output_path=str(out_path),
                mount=mount,
            ),
            shell=True,  # noqa: S602
            check=False,
            executable=executable,
            capture_output=True,
        )

    # Ligand metrics
    out_path = Path(outdir) / f"{name}_ligand.json"
    if not out_path.exists():
        subprocess.run(
            OST_COMPARE_LIGAND.format(
                model_file=pred,
                reference_file=reference,
                output_path=str(out_path),
                mount=mount,
            ),
            shell=True,  # noqa: S602
            check=False,
            executable=executable,
            capture_output=True,
        )


def find_predictions(pred_dir: Path, target_name: str, num_samples: int) -> list[Path]:
    """Find prediction CIF files for a given target, trying common naming patterns."""
    results = []
    for model_idx in range(num_samples):
        # Try Boltz naming: {name}_model_{idx}.cif
        candidates = [
            pred_dir / target_name / f"{target_name}_model_{model_idx}.cif",
            pred_dir / target_name.lower() / f"{target_name.lower()}_model_{model_idx}.cif",
            pred_dir / target_name.upper() / f"{target_name.upper()}_model_{model_idx}.cif",
        ]
        for c in candidates:
            if c.exists():
                results.append(c)
                break
    return results


def find_reference(ref_dir: Path, target_name: str) -> Path | None:
    """Find reference CIF file for a target."""
    candidates = [
        ref_dir / f"{target_name}.cif",
        ref_dir / f"{target_name.lower()}.cif",
        ref_dir / f"{target_name}.cif.gz",
        ref_dir / f"{target_name.lower()}.cif.gz",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenStructure evaluation on Boltz predictions"
    )
    parser.add_argument("predictions", type=Path, help="Directory with Boltz predictions")
    parser.add_argument("references", type=Path, help="Directory with reference CIF files")
    parser.add_argument("outdir", type=Path, help="Output directory for eval JSONs")
    parser.add_argument("--mount", type=str, required=True,
                        help="Docker mount path (common parent of all paths)")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of diffusion samples per target")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="Max parallel Docker evaluations")
    parser.add_argument("--executable", type=str, default="/bin/bash")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Discover targets from predictions directory
    pred_dirs = sorted([
        d for d in args.predictions.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    print(f"Found {len(pred_dirs)} prediction directories")

    # Build evaluation tasks
    tasks = []
    skipped = []
    for pred_dir in pred_dirs:
        target_name = pred_dir.name
        ref_path = find_reference(args.references, target_name)
        if ref_path is None:
            skipped.append(target_name)
            continue

        preds = find_predictions(args.predictions, target_name, args.num_samples)
        for pred_path in preds:
            model_idx = pred_path.stem.split("_model_")[-1]
            eval_name = f"{target_name.lower()}_model_{model_idx}"
            tasks.append((eval_name, str(pred_path), str(ref_path)))

    if skipped:
        print(f"Skipped {len(skipped)} targets (no reference found): {skipped[:5]}...")

    print(f"Running {len(tasks)} evaluations with {args.max_workers} workers")

    # Run first task synchronously (ensures Docker image is downloaded)
    if tasks:
        name, pred, ref = tasks[0]
        evaluate_structure(name, pred, ref, str(args.outdir), args.mount, args.executable)
        remaining = tasks[1:]
    else:
        remaining = []

    # Run remaining in parallel
    with concurrent.futures.ThreadPoolExecutor(args.max_workers) as executor:
        futures = []
        for name, pred, ref in remaining:
            future = executor.submit(
                evaluate_structure, name, pred, ref,
                str(args.outdir), args.mount, args.executable,
            )
            futures.append(future)

        with tqdm(total=len(futures)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

    print(f"Done. Results in {args.outdir}")


if __name__ == "__main__":
    main()
