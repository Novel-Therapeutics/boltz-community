"""Refresh reviewed integrated-source identities after inspecting local changes.

Runtime activation verifies these hashes and never refreshes them itself.
The original numerical source pin is immutable under this command.
"""

import hashlib
import json
from pathlib import Path

import tomllib


def main():
    repository = Path(__file__).resolve().parents[1]
    fork = repository / "src/boltz"
    contract_path = fork / "upstream/contract.json"
    current = json.loads(contract_path.read_text())
    root = fork / "_anthropic_engine/engine"
    reference = fork / "_anthropic_engine/reference"
    package = repository / "src/boltz2_opt"

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for relative, wanted in current["stock_files"].items():
        if digest(reference / relative) != wanted:
            raise ValueError(f"Reference numerical implementation changed: {relative}")
    current["fork_version"] = tomllib.loads(
        (repository / "pyproject.toml").read_text()
    )["project"]["version"]
    current["engine_files"] = {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    }
    current["engine_package_files"] = {
        path.name: digest(path) for path in sorted(package.iterdir()) if path.is_file()
    }
    current["fork_files"] = {
        path.relative_to(fork).as_posix(): digest(path)
        for path in sorted(fork.rglob("*.py"))
    }
    current["engine_adjustments"] = [
        "Native package CLI delegates to the integrated fork launcher.",
        "Installed engine assets and pyproject resolve inside this distribution.",
        "Source proof identifies the fork and packaged numerical modules separately.",
        "The core pin, flags, profiles, kernels and numerical source are preserved.",
        "Worker failure accounting handles the fork writer's skipped-batch exception.",
        "Overlapped output preserves the fork's optional confidence and structure checks.",
        "The off route accounts for named writer failures without abandoning successful records.",
    ]
    contract_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    print("Recorded integrated engine, assets and fork source identities")


if __name__ == "__main__":
    main()
