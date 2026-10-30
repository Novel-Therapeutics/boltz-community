"""Bounded, invocation-owned caches for eager Boltz-2 denoising inference."""

import functools
import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)
MAX_CACHE_BYTES = 128 * 1024**2
_MISSING = object()


def cache_enabled():
    """Parse once at CLI startup, and once per direct sampling invocation."""
    setting = os.environ.get("BOLTZ_DENOISER_CACHE", "off").strip().lower()
    if setting in ("off", "0", "false"):
        return False
    if setting in ("on", "1", "true"):
        return True
    raise ValueError("BOLTZ_DENOISER_CACHE must be on/off, true/false, or 1/0")


def _eligible(module, x):
    if torch.compiler.is_compiling():
        return False
    return (
        not module.training
        and not torch.is_grad_enabled()
        and x.is_cuda
        and (
            not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") == torch.bfloat16
        )
    )


def active(cache, module):
    """Do not take Python cache paths into training or compiled graphs."""
    if torch.compiler.is_compiling():
        return None
    if cache is None or module.training or torch.is_grad_enabled() or cache.closed:
        return None
    return cache


def _tensor_key(tensor):
    try:
        version = tensor._version
    except RuntimeError:
        return None
    return (
        tensor.data_ptr(),
        version,
        tensor.shape,
        tensor.stride(),
        tensor.dtype,
        tensor.device,
    )


def _storages(tensors):
    # Count shared views once, including references retained for identity safety.
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    return storages


@dataclass(eq=False)
class _ModulePlan:
    modules: tuple
    owners: tuple
    bindings: tuple
    settings: tuple
    parameters: tuple
    buffers: tuple

    @classmethod
    def capture(cls, modules):
        owners = tuple(child for m in modules for child in m.modules())
        return cls(
            modules,
            owners,
            tuple(
                tuple(
                    tuple(getattr(m, attr).items())
                    for attr in ("_parameters", "_buffers", "_modules")
                )
                for m in owners
            ),
            tuple(
                (m.__dict__.get("eps"), m.__dict__.get("normalized_shape"))
                for m in owners
            ),
            tuple(p for m in modules for p in m.parameters()),
            tuple(b for m in modules for b in m.buffers()),
        )

    def matches(self, modules):
        if self.modules != modules:
            return False
        for owner, bindings, settings in zip(self.owners, self.bindings, self.settings):
            for attr, expected in zip(
                ("_parameters", "_buffers", "_modules"), bindings
            ):
                current = getattr(owner, attr)
                if len(current) != len(expected) or any(
                    current.get(k, _MISSING) is not v for k, v in expected
                ):
                    return False
            if (
                owner.__dict__.get("eps"),
                owner.__dict__.get("normalized_shape"),
            ) != settings:
                return False
        return True


@dataclass
class _Entry:
    signature: tuple
    value: object
    references: tuple
    storages: dict
    value_versions: tuple
    plan: _ModulePlan


class SamplingCache:
    """A sampler-local cache; no module attributes or global tensor state."""

    def __init__(self, max_bytes=MAX_CACHE_BYTES):
        self.max_bytes = max_bytes
        self.entries = {}
        self.roots = ()
        self._storage_refs = {}
        self.bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.builds = 0
        self.declines = 0
        self.declined = set()
        self.declined_calls = 0
        self.plan_builds = 0
        self.closed = False

    def _retain(self, storages):
        for key, size in storages.items():
            if key not in self._storage_refs:
                self.bytes += size
                self._storage_refs[key] = 0
            self._storage_refs[key] += 1
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    def _release(self, storages):
        for key, size in storages.items():
            self._storage_refs[key] -= 1
            if not self._storage_refs[key]:
                del self._storage_refs[key]
                self.bytes -= size

    def own_inputs(self, kwargs):
        """Copy only the small invariant inputs, with usable version counters."""
        if self.closed or self.roots:
            raise RuntimeError("SamplingCache input ownership is single-use")
        trunk, inputs = kwargs["s_trunk"], kwargs["s_inputs"]
        cond = kwargs["diffusion_conditioning"]
        atom = cond["c"]
        size = sum(t.numel() * t.element_size() for t in [trunk, inputs])
        size += atom.numel() * 4
        if size > self.max_bytes:
            return None
        with torch.inference_mode(False), torch.no_grad():
            trunk = trunk.detach().clone()
            inputs = inputs.detach().clone()
            atom = atom.detach().to(dtype=torch.float32, copy=True)
        self.roots = (trunk, inputs, atom)
        self._retain(_storages(self.roots))
        return {
            **kwargs,
            "s_trunk": trunk,
            "s_inputs": inputs,
            "diffusion_conditioning": {**cond, "c": atom},
            "_inference_cache": self,
        }

    def get(self, role, inputs, modules, build):
        """Reuse only versioned inputs/weights in the same precision and shape."""
        if torch.compiler.is_compiling():
            return build()
        if self.closed or torch.is_grad_enabled() or any(m.training for m in modules):
            return build()
        if role in self.declined:
            self.declined_calls += 1
            return build()
        entry = self.entries.get(role)
        if entry is not None and entry.plan.matches(modules):
            plan = entry.plan
        else:
            plan = _ModulePlan.capture(modules)
            self.plan_builds += 1
        references = (*inputs, *plan.parameters, *plan.buffers)
        keys = tuple(_tensor_key(t) for t in references)
        if any(k is None for k in keys):
            return build()
        device = inputs[0].device.type
        autocast = torch.is_autocast_enabled(device)
        signature = (
            plan,
            keys,
            autocast,
            torch.get_autocast_dtype(device) if autocast else None,
            torch.get_float32_matmul_precision(),
        )
        if entry is not None:
            old_outputs = (
                (entry.value,)
                if isinstance(entry.value, torch.Tensor)
                else tuple(entry.value)
            )
            if entry.signature == signature and entry.value_versions == tuple(
                _tensor_key(t) for t in old_outputs
            ):
                self.hits += 1
                return entry.value
            del old_outputs
            self._release(entry.storages)
            del self.entries[role]
            del entry
        self.builds += 1
        with torch.inference_mode(False), torch.no_grad():
            value = build()
        outputs = (value,) if isinstance(value, torch.Tensor) else tuple(value)
        value_versions = tuple(_tensor_key(t) for t in outputs)
        storages = _storages((*references, *outputs))
        additional = sum(n for k, n in storages.items() if k not in self._storage_refs)
        if (
            all(k is not None for k in value_versions)
            and additional + self.bytes <= self.max_bytes
        ):
            self._retain(storages)
            self.entries[role] = _Entry(
                signature, value, references, storages, value_versions, plan
            )
        else:
            self.declines += 1
            self.declined.add(role)
        return value

    def single_prefix(self, module, trunk, inputs, multiplicity):
        return self.get(
            ("single", module, multiplicity),
            (trunk, inputs),
            (module.norm_single, module.single_embed),
            lambda: module.single_embed(
                module.norm_single(
                    torch.cat(
                        (
                            trunk.repeat_interleave(multiplicity, 0),
                            inputs.repeat_interleave(multiplicity, 0),
                        ),
                        dim=-1,
                    )
                )
            ),
        )

    def repeat_conditioning(self, conditioning, multiplicity):
        return self.get(
            ("atom", multiplicity),
            (conditioning,),
            (),
            lambda: conditioning.repeat_interleave(multiplicity, 0),
        )

    def modulation(self, module, conditioning):
        def build():
            normalized = module.s_norm(conditioning)
            return torch.sigmoid(module.s_scale(normalized)), module.s_bias(normalized)

        return self.get(
            ("adaln", module, conditioning.shape),
            (conditioning,),
            (module,),
            build,
        )

    def gate(self, module, conditioning):
        return self.get(
            ("gate", module, conditioning.shape),
            (conditioning,),
            (module,),
            lambda: module(conditioning),
        )

    def clear(self):
        self.entries.clear()
        self.declined.clear()
        self.roots = ()
        self._storage_refs.clear()
        self.bytes = 0
        self.closed = True


def cache_sampling(function):
    """Cache only the stock eager Boltz-2 score model for one sample() call."""

    @functools.wraps(function)
    def scoped(module, atom_mask, *args, **kwargs):
        enabled = cache_enabled()
        if not enabled or not _eligible(module, atom_mask):
            return function(module, atom_mask, *args, **kwargs)
        from boltz.model.modules.diffusionv2 import DiffusionModule

        # Compiled wrappers and custom score models retain their original API.
        if (
            type(module.score_model) is not DiffusionModule
            or module.score_model.training
        ):
            return function(module, atom_mask, *args, **kwargs)
        if not all(
            k in kwargs for k in ("s_trunk", "s_inputs", "diffusion_conditioning")
        ):
            return function(module, atom_mask, *args, **kwargs)
        budget = MAX_CACHE_BYTES
        if atom_mask.is_cuda:
            budget = min(budget, torch.cuda.mem_get_info(atom_mask.device)[0] // 8)
        cache = SamplingCache(budget)
        try:
            owned = cache.own_inputs(kwargs)
            return function(
                module, atom_mask, *args, **(kwargs if owned is None else owned)
            )
        finally:
            logger.debug(
                "Denoiser cache: hits=%d builds=%d declines=%d "
                "declined_calls=%d plans=%d peak_bytes=%d",
                cache.hits,
                cache.builds,
                cache.declines,
                cache.declined_calls,
                cache.plan_builds,
                cache.peak_bytes,
            )
            cache.clear()

    return scoped
