"""Keep the engine's result and preprocessing state out of native workspaces."""

from __future__ import annotations

import json
from pathlib import Path

RUN_MARKER = ".boltz2-opt-run.json"


class OutputConflict(RuntimeError):
    """The requested output root already belongs to another invocation."""


def reserve_engine_output(out_dir: str | Path, mode: str) -> Path:
    """Claim a fresh prediction root before staging or dispatching the engine."""
    from .contract import CONTRACT_PATH, digest

    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / RUN_MARKER
    try:
        claim = marker.open("x", encoding="utf-8")
    except FileExistsError as error:
        message = f"{root} already has an engine run. Choose a fresh output directory."
        raise OutputConflict(message) from error

    try:
        with claim:
            occupied = next(
                (
                    entry.name
                    for entry in root.iterdir()
                    if entry.name in {"by_seed", "_kit"}
                    or entry.name.startswith("boltz_results_")
                    or entry.name.endswith("_worker.log")
                ),
                None,
            )
            if occupied is not None:
                message = (
                    f"{root} already contains prediction state ({occupied}). "
                    "Choose a fresh output directory."
                )
                raise OutputConflict(message)
            json.dump(
                {"mode": mode, "source_contract_sha256": digest(CONTRACT_PATH)}, claim
            )
            claim.write("\n")
    except BaseException:
        marker.unlink()
        raise
    return root


def prepare_native_output(root: Path, stem: str) -> Path:
    """Set up a native result path while honoring an engine's root claim."""
    marker = root / RUN_MARKER
    message = (
        f"{root} belongs to a boltz2-opt run. "
        "Choose a separate output directory for boltz predict."
    )
    if marker.exists():
        raise OutputConflict(message)

    # Publish the result directory before rechecking the marker. If the
    # engine marks the root first, we stop. If this directory appears first,
    # the engine's existing-result check prevents it from starting.
    result = root / f"boltz_results_{stem}"
    try:
        result.mkdir(parents=True)
    except FileExistsError:
        if not result.is_dir():
            raise
        created = False
    else:
        created = True

    if marker.exists():
        if created:
            try:
                result.rmdir()
            except OSError:
                # Another native caller or external writer may hold the path.
                pass
        raise OutputConflict(message)
    return result
