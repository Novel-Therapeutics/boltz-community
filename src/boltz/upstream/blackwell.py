"""Blackwell launch adaptations for the pinned Anthropic provider sources."""

from __future__ import annotations

import hashlib
from pathlib import Path

EXACTLN_FILES = {
    "__init__.py": "fd492146245faeb555f0bef684ceba35dac48006f13c140a0ce423f924fbd14b",
    "exactln_fwd.cu": "8283eb74c962c96f8eb4b5f082cf5c7eaeb4ecdf3f2ebccd70f2424d89d79df1",
}
OPM_CONFIG = "community_sm120"


def admits_layernorm(package, cc, torch_version):
    """Admit the tested source on sm_120, retaining per-call bit comparisons."""
    if tuple(cc) != (12, 0) or str(torch_version).split("+")[0] != "2.12.0":
        return False
    if getattr(package, "TORCH_PIN", None) != "2.12.0":
        return False
    root = Path(package.__file__).parent
    return all(
        (root / filename).is_file()
        and hashlib.sha256((root / filename).read_bytes()).hexdigest() == expected
        for filename, expected in EXACTLN_FILES.items()
    )


def configure_opm(provider, cc):
    """Keep the fused prologue and projection with an sm_120-sized shared tile."""
    if tuple(cc) != (12, 0):
        return None
    current, origin = provider.cfg_for("mask_norm", 128)
    if origin != "default" or provider.cfg_name(current) != "f1":
        return None
    config = {**current, "BJ": 4, "num_stages": 2}
    existing = provider.CFG_VARIANTS.get(OPM_CONFIG)
    if existing is not None and existing != config:
        raise RuntimeError("Blackwell OPM configuration name is already in use")
    provider.CFG_VARIANTS[OPM_CONFIG] = config
    return OPM_CONFIG
