"""Experimental invocation-owned CUDA graphs for deterministic denoiser calls."""

import functools
import logging
import os
import time
from dataclasses import dataclass

import torch

from boltz.model.layers.denoiser_inference import (
    SamplingCache,
    _ModulePlan,
    _tensor_key,
)

logger = logging.getLogger(__name__)
MAX_GRAPH_BYTES = 8 * 1024**3


def graph_enabled():
    value = os.environ.get("BOLTZ_DENOISER_GRAPH", "off").strip().lower()
    if value in ("off", "false", "0"):
        return False
    if value in ("on", "true", "1"):
        return True
    raise ValueError("BOLTZ_DENOISER_GRAPH must be on/off, true/false, or 1/0")


def _eligible(module, mask):
    if torch.compiler.is_compiling():
        return False
    from boltz.model.modules.diffusionv2 import DiffusionModule

    return (
        not module.training
        and not torch.is_grad_enabled()
        and mask.is_cuda
        and type(module.score_model) is DiffusionModule
        and not module.score_model.training
        and torch.get_default_dtype() == torch.float32
        and (
            not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") == torch.bfloat16
        )
    )


def _tree_key(value, references, cache_roles):
    if isinstance(value, torch.Tensor):
        references.append(value)
        key = _tensor_key(value)
        if key is None:
            # Inference-mode input tensors are read live by the graph. No
            # computed invariant from them is stored here.
            key = (
                value.data_ptr(),
                tuple(value.shape),
                value.stride(),
                value.dtype,
                value.device,
            )
        return (id(value), key)
    if isinstance(value, SamplingCache):
        roles = cache_roles.setdefault(id(value), tuple(value.entries))
        return (
            id(value),
            value.closed,
            tuple(
                (
                    role,
                    _tree_key(value.entries[role].value, references, cache_roles)
                    if role in value.entries
                    else None,
                )
                for role in roles
            ),
        )
    if isinstance(value, dict):
        return tuple(
            (k, _tree_key(v, references, cache_roles)) for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return (
            type(value),
            tuple(_tree_key(v, references, cache_roles) for v in value),
        )
    if isinstance(value, functools.partial):
        return (
            value.func,
            _tree_key(value.args, references, cache_roles),
            _tree_key(value.keywords, references, cache_roles),
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return (type(value), id(value))


@dataclass
class _Entry:
    graph: object
    stream: object
    coords: torch.Tensor
    sigma: torch.Tensor
    output: torch.Tensor
    signature: tuple
    references: tuple
    pool_bytes: int
    cache_roles: dict


class DenoiserGraphs:
    """Graphs live for one sample() and never replace stochastic sampler work."""

    def __init__(self, sampler, max_bytes=MAX_GRAPH_BYTES):
        self.sampler = sampler
        self.plan = _ModulePlan.capture((sampler.score_model,))
        self.entries = {}
        self.declined = set()
        self.max_bytes = max_bytes
        self.pool_bytes = 0
        self.peak_pool_bytes = 0
        self.captures = 0
        self.replays = 0
        self.eager_calls = 0
        self.invalidations = 0
        self.capture_seconds = 0.0
        self.reason = None
        self.closed = False

    def _signature(self, kwargs, cache_roles=None):
        model = self.sampler.score_model
        if not self.plan.matches((model,)):
            self.plan = _ModulePlan.capture((model,))
        if any(
            m.training or m._forward_hooks or m._forward_pre_hooks
            for m in self.plan.owners
        ):
            return None, ()
        weights = tuple(
            _tensor_key(p) for p in (*self.plan.parameters, *self.plan.buffers)
        )
        if any(k is None for k in weights):
            return None, ()
        references = []
        tree = _tree_key(kwargs, references, {} if cache_roles is None else cache_roles)
        if any(t.device.type != "cuda" and torch.is_inference(t) for t in references):
            # CPU operations are not replayed by a CUDA graph, so an
            # unversioned CPU input cannot be treated as a live graph input.
            return None, ()
        amp = torch.is_autocast_enabled("cuda")
        flags = tuple(
            (getattr(m, "use_flash_attn", None), getattr(m, "use_kernels", None))
            for m in self.plan.owners
        )
        return (
            self.plan,
            weights,
            tree,
            flags,
            amp,
            torch.get_autocast_dtype("cuda") if amp else None,
            torch.get_float32_matmul_precision(),
        ), tuple(references)

    def _eager(self, coords, sigma, kwargs):
        self.eager_calls += 1
        return self.sampler.preconditioned_network_forward(
            coords, sigma, network_condition_kwargs=kwargs
        )

    def _projected_bytes(self, coords, kwargs):
        cond = kwargs["diffusion_conditioning"]
        casts = sum(
            t.numel() * 4
            for t in cond.values()
            if isinstance(t, torch.Tensor) and t.dtype != torch.float32
        )
        b, n = coords.shape[0], kwargs["s_trunk"].shape[1]
        heads = self.sampler.score_model.token_transformer.layers[
            0
        ].pair_bias_attn.num_heads
        atom_width = cond["c"].shape[-1]
        return (
            casts
            + 3 * b * heads * n * n * 4
            + 16 * coords.shape[1] * b * atom_width * 4
        )

    def _capture(self, coords, sigma, kwargs, eager):
        device = coords.device
        free, _ = torch.cuda.mem_get_info(device)
        projected = self._projected_bytes(coords, kwargs)
        limit = min(self.max_bytes, free // 4)
        if self.pool_bytes + projected > limit or 2 * projected + 512 * 1024**2 > free:
            return None, "headroom"
        stream = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        static_coords = torch.empty_strided(
            coords.shape, coords.stride(), device=device, dtype=coords.dtype
        )
        static_coords.copy_(coords)
        static_sigma = torch.full((coords.shape[0],), sigma, device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        started = time.perf_counter()
        # Warmup/capture are engineering work, not additional RNG draws in the
        # scientific request. Reject a denoiser that consumes random numbers.
        with torch.random.fork_rng(devices=[device]):
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.sampler.preconditioned_network_forward(
                        static_coords, static_sigma, network_condition_kwargs=kwargs
                    )
            stream.synchronize()
            if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
                cuda_rng, torch.cuda.get_rng_state(device)
            ):
                return None, "random_denoiser"
            with torch.cuda.graph(graph, stream=stream):
                output = self.sampler.preconditioned_network_forward(
                    static_coords, static_sigma, network_condition_kwargs=kwargs
                )
            torch.cuda.current_stream(device).wait_stream(stream)
            graph.replay()
            if not torch.equal(
                output.contiguous().view(torch.uint8),
                eager.contiguous().view(torch.uint8),
            ):
                return None, "capture_parity"
        segments = torch.cuda.memory_snapshot()
        if not segments or "segment_pool_id" not in segments[0]:
            return None, "pool_accounting_unavailable"
        size = sum(
            s["total_size"]
            for s in segments
            if tuple(s["segment_pool_id"]) == tuple(graph.pool())
        )
        if size + self.pool_bytes > limit:
            return None, "pool_budget"
        cache_roles = {}
        signature, references = self._signature(kwargs, cache_roles)
        if signature is None:
            return None, "unsupported_model"
        self.captures += 1
        self.capture_seconds += time.perf_counter() - started
        self.pool_bytes += size
        self.peak_pool_bytes = max(self.peak_pool_bytes, self.pool_bytes)
        return _Entry(
            graph,
            stream,
            static_coords,
            static_sigma,
            output,
            signature,
            references,
            size,
            cache_roles,
        ), None

    def run(self, coords, sigma, kwargs):
        shape = (
            tuple(coords.shape),
            coords.stride(),
            coords.dtype,
            coords.device,
            kwargs["multiplicity"],
        )
        if self.closed or self.reason is not None or shape in self.declined:
            return self._eager(coords, sigma, kwargs)
        if torch.is_grad_enabled() or not isinstance(sigma, float):
            return self._eager(coords, sigma, kwargs)
        entry = self.entries.get(shape)
        signature, references = self._signature(
            kwargs, entry.cache_roles if entry is not None else None
        )
        if signature is None:
            self.reason = "unsupported_model"
            return self._eager(coords, sigma, kwargs)
        if entry is not None and entry.signature != signature:
            torch.cuda.synchronize(coords.device)
            self.pool_bytes -= entry.pool_bytes
            del self.entries[shape]
            del entry
            self.invalidations += 1
        entry = self.entries.get(shape)
        if entry is not None:
            entry.coords.copy_(coords)
            entry.sigma.fill_(sigma)
            entry.graph.replay()
            self.replays += 1
            return entry.output
        eager = self._eager(coords, sigma, kwargs)
        try:
            entry, refusal = self._capture(coords, sigma, kwargs, eager)
        except RuntimeError as error:
            if (
                isinstance(error, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(error).lower()
            ):
                raise
            entry, refusal = None, "capture_error: " + str(error)
        if entry is not None:
            self.entries[shape] = entry
        else:
            self.declined.add(shape)
            logger.warning("Denoiser graph declined: %s", refusal)
        return eager

    def close(self):
        for device in {entry.coords.device for entry in self.entries.values()}:
            torch.cuda.synchronize(device)
        self.entries.clear()
        self.declined.clear()
        self.pool_bytes = 0
        self.closed = True


def graph_sampling(function):
    @functools.wraps(function)
    def scoped(module, atom_mask, *args, **kwargs):
        enabled = graph_enabled()
        if not enabled or not _eligible(module, atom_mask):
            return function(module, atom_mask, *args, **kwargs)
        runner = DenoiserGraphs(module)
        try:
            return function(module, atom_mask, *args, _denoiser_graph=runner, **kwargs)
        finally:
            logger.debug(
                "Denoiser graph: captures=%d replays=%d eager=%d "
                "invalidations=%d pool_peak=%d capture_s=%.6f reason=%s",
                runner.captures,
                runner.replays,
                runner.eager_calls,
                runner.invalidations,
                runner.peak_pool_bytes,
                runner.capture_seconds,
                runner.reason,
            )
            runner.close()

    return scoped
