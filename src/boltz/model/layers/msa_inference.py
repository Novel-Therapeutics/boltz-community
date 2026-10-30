"""Reuse invariant work inside one Boltz-2 MSA inference call.

The mask counts are forward-local intermediates, never a process-wide tensor
cache. Inference-mode masks are copied to versioned tensors for the MSA forward;
version counters invalidate cached counts if a layer edits its mask in place.
"""

import functools
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import Tensor

_mask_counts: ContextVar[dict | None] = ContextVar("msa_mask_counts", default=None)


def _admitted(x: Tensor, *, training: bool = False) -> bool:
    """Keep admission criteria in one place, outside compiled graphs."""
    if torch.compiler.is_compiling():
        return False
    return (
        not training
        and not torch.is_grad_enabled()
        and x.is_cuda
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )


def enabled(x: Tensor) -> bool:
    """Whether this call is scoped, gradient-free CUDA bf16 inference."""
    return _admitted(x) and _mask_counts.get() is not None


@contextmanager
def inference_scope(x: Tensor, *, training: bool):
    """Bound reuse to one MSA forward and restore nested scopes on all exits."""
    token = _mask_counts.set({} if _admitted(x, training=training) else None)
    try:
        yield
    finally:
        _mask_counts.reset(token)


def msa_forward(function):
    """Enable reuse only in the Boltz-2 MSA module's inference forward."""

    @functools.wraps(function)
    def scoped(module, z, *args, **kwargs):
        # Keep ContextVar operations out of compiled training/CPU forwards.
        if not _admitted(z, training=module.training):
            return function(module, z, *args, **kwargs)
        with inference_scope(z, training=module.training):
            return function(module, z, *args, **kwargs)

    return scoped


def cast_activation(x: Tensor) -> Tensor:
    """Perform autocast's fp32-to-bf16 conversion once before a chunk loop."""
    if enabled(x) and x.dtype == torch.float32:
        return x.to(torch.bfloat16)
    return x


def versioned_mask(mask: Tensor) -> Tensor:
    """Own a versioned copy when inference tensors cannot track mutations."""
    if enabled(mask) and torch.is_inference(mask):
        with torch.inference_mode(False), torch.no_grad():
            return mask.clone()
    return mask


def mask_counts(raw_mask: Tensor, mask: Tensor) -> Tensor:
    """Compute the original chunked denominator, reusing it within this forward."""
    cache = _mask_counts.get() if enabled(mask) else None
    key = None
    if cache is not None:
        try:
            version = raw_mask._version
        except RuntimeError:
            # Direct callers can supply unversioned masks; never cache them.
            cache = None
    if cache is not None:
        key = (
            id(raw_mask),
            version,
            raw_mask.data_ptr(),
            raw_mask.shape,
            raw_mask.stride(),
            mask.dtype,
            mask.device,
        )
        if key in cache:
            return cache[key][1]
    # An empty first block sums to zero, giving a clamped denominator of one.
    count = (mask[:, :64, None, :] * mask[:, :64, :, None]).sum(1)
    for i in range(64, mask.shape[1], 64):
        count += (mask[:, i : i + 64, None, :] * mask[:, i : i + 64, :, None]).sum(1)
    count = count.clamp(min=1)
    if cache is not None:
        cache[key] = (raw_mask, count)  # Keep the input alive until scope cleanup.
    return count
