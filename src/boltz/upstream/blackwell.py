"""Blackwell launch adaptations for the pinned Anthropic provider sources."""

from __future__ import annotations

import hashlib
from pathlib import Path

EXACTLN_FILES = {
    "__init__.py": "fd492146245faeb555f0bef684ceba35dac48006f13c140a0ce423f924fbd14b",
    "exactln_fwd.cu": "8283eb74c962c96f8eb4b5f082cf5c7eaeb4ecdf3f2ebccd70f2424d89d79df1",
}
OPM_CONFIG = "community_sm120"
EXACT_CELL_FILES = {
    "dit": "8a424aa702c5d62c4dcffd6b20ec0d1f2f38748a6fba0842a57d2558fae60e64",
    "trans2x": "04b5d8798d90243705c056e2aaf00b894f9f4f4ca4a404af7d7fbb07e02712fd",
    "pwa2x": "0331762dc9e9dc31d5381bcac282e9a1544e291207f0257acf634ff0b9c7a701",
}


def stock_triangle_bias_core(module, x_ln, core):
    """Keep cuBLAS's narrow bias projection while the other block stages fuse."""
    # The inherited fused projection changes a few BF16 bias values on sm_120.
    # Match the stock Linear's GEMM shape and rounding, in the attention frame.
    bias = module.linear(x_ln).movedim(-1, -3).unsqueeze(-4)
    if x_ln.dim() == 3:
        bias = bias.unsqueeze(0)

    def attention(q, k, v, _provider_bias, mask, scale):
        return core(q, k, v, bias, mask, scale)

    return attention


def admits_exact_cell(module, cell, cc, torch_version, cuda_version):
    """Qualify only the probed Blackwell source/stack; runtime locks still apply."""
    if (
        tuple(cc) != (12, 0)
        or str(torch_version).split("+")[0] != "2.12.0"
        or cuda_version != "13.0"
    ):
        return False
    source = Path(module.__file__)
    return (
        cell in EXACT_CELL_FILES
        and source.is_file()
        and hashlib.sha256(source.read_bytes()).hexdigest() == EXACT_CELL_FILES[cell]
    )


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
