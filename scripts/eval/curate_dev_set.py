"""Curate a small, representative dev benchmark from Boltz eval data.

Parses all input YAMLs and selects targets stratified by:
  - Type: protein-only, protein-ligand, protein-nucleic-acid, multi-chain
  - Size: small (<200 residues), medium (200-500), large (500-1000)

Excludes very large targets (>1000 total residues) to keep runtime fast.

Usage:
    python scripts/eval/curate_dev_set.py \
        ~/boltz-benchmark/data/boltz_results_final/inputs/test/boltz/queries/ \
        --num-targets 25 \
        --max-residues 1000 \
        --output dev_targets.txt
"""

import argparse
import sys
from pathlib import Path

import yaml


def parse_yaml_target(yaml_path: Path) -> dict | None:
    """Extract target metadata from a Boltz input YAML."""
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except Exception:
        return None

    if not data or "sequences" not in data:
        return None

    total_residues = 0
    n_protein_chains = 0
    n_ligands = 0
    n_nucleic = 0
    has_affinity = False

    for entry in data["sequences"]:
        if "protein" in entry:
            prot = entry["protein"]
            seq = prot.get("sequence", "")
            # id can be a string or list — each id is a copy of the chain
            ids = prot.get("id", "A")
            n_copies = len(ids) if isinstance(ids, list) else 1
            total_residues += len(seq) * n_copies
            n_protein_chains += n_copies

        elif "rna" in entry or "dna" in entry:
            key = "rna" if "rna" in entry else "dna"
            seq = entry[key].get("sequence", "")
            ids = entry[key].get("id", "X")
            n_copies = len(ids) if isinstance(ids, list) else 1
            total_residues += len(seq) * n_copies
            n_nucleic += len(ids) if isinstance(ids, list) else 1

        elif "ligand" in entry:
            ids = entry["ligand"].get("id", "L")
            n_ligands += len(ids) if isinstance(ids, list) else 1

    if "properties" in data:
        for prop in data["properties"]:
            if "affinity" in prop:
                has_affinity = True

    # Classify target type
    if n_nucleic > 0:
        target_type = "nucleic"
    elif n_ligands > 0:
        target_type = "protein-ligand"
    elif n_protein_chains > 1:
        target_type = "multi-chain"
    else:
        target_type = "protein-only"

    # Size bin
    if total_residues < 200:
        size_bin = "small"
    elif total_residues < 500:
        size_bin = "medium"
    elif total_residues <= 1000:
        size_bin = "large"
    else:
        size_bin = "xlarge"

    return {
        "name": yaml_path.stem,
        "total_residues": total_residues,
        "n_protein_chains": n_protein_chains,
        "n_ligands": n_ligands,
        "n_nucleic": n_nucleic,
        "has_affinity": has_affinity,
        "target_type": target_type,
        "size_bin": size_bin,
    }


def curate(targets: list[dict], num_targets: int, max_residues: int, seed: int) -> list[dict]:
    """Select a stratified subset of targets."""
    import random

    rng = random.Random(seed)

    # Filter out targets that are too large
    eligible = [t for t in targets if t["total_residues"] <= max_residues]

    # Group by (type, size_bin)
    groups: dict[tuple[str, str], list[dict]] = {}
    for t in eligible:
        key = (t["target_type"], t["size_bin"])
        groups.setdefault(key, []).append(t)

    # Sort each group by residue count for determinism
    for key in groups:
        groups[key].sort(key=lambda t: (t["total_residues"], t["name"]))

    # Print available groups
    print(f"\nEligible targets: {len(eligible)} (filtered from {len(targets)})")
    print(f"Max residues: {max_residues}")
    print(f"\nGroups:")
    for (ttype, sbin), members in sorted(groups.items()):
        print(f"  {ttype:20s} {sbin:8s}: {len(members)} targets")

    # Stratified selection: distribute budget across groups proportionally,
    # ensuring at least 1 from each group
    selected = []
    remaining_budget = num_targets

    # First pass: pick 1 from each group
    for key in sorted(groups.keys()):
        if remaining_budget <= 0:
            break
        pick = rng.choice(groups[key])
        selected.append(pick)
        groups[key].remove(pick)
        remaining_budget -= 1

    # Second pass: distribute remaining budget proportionally
    if remaining_budget > 0:
        all_remaining = []
        for members in groups.values():
            all_remaining.extend(members)
        rng.shuffle(all_remaining)
        selected.extend(all_remaining[:remaining_budget])

    # Sort final selection by type then size for readability
    selected.sort(key=lambda t: (t["target_type"], t["total_residues"]))
    return selected


def main():
    parser = argparse.ArgumentParser(description="Curate dev benchmark target set")
    parser.add_argument("queries_dir", type=Path, help="Path to queries/ directory with YAML files")
    parser.add_argument("--num-targets", type=int, default=25, help="Number of targets to select")
    parser.add_argument("--max-residues", type=int, default=1000,
                        help="Max total residues per target (excludes very large)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selection")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file (one target name per line). Prints to stdout if not set.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed target info")
    args = parser.parse_args()

    if not args.queries_dir.is_dir():
        print(f"Error: {args.queries_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    yamls = sorted(args.queries_dir.glob("*.yaml"))
    print(f"Scanning {len(yamls)} YAML files...")

    targets = []
    for yp in yamls:
        meta = parse_yaml_target(yp)
        if meta:
            targets.append(meta)

    print(f"Parsed {len(targets)} targets")

    selected = curate(targets, args.num_targets, args.max_residues, args.seed)

    print(f"\nSelected {len(selected)} targets:")
    print(f"{'Name':30s} {'Type':20s} {'Size':8s} {'Residues':>8s} {'Chains':>6s} {'Lig':>4s} {'Nuc':>4s}")
    print("-" * 84)
    for t in selected:
        print(f"{t['name']:30s} {t['target_type']:20s} {t['size_bin']:8s} "
              f"{t['total_residues']:8d} {t['n_protein_chains']:6d} {t['n_ligands']:4d} {t['n_nucleic']:4d}")

    names = [t["name"] for t in selected]

    if args.output:
        args.output.write_text("\n".join(names) + "\n")
        print(f"\nTarget list written to {args.output}")
    else:
        print(f"\nTarget names:")
        for n in names:
            print(n)


if __name__ == "__main__":
    main()
