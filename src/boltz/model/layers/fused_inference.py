"""Opt-in, invocation-owned Anthropic fused inference kernels.

The stock sampler, guidance, chunk indexing and attention-backend selection stay
in the fork. Packed values never survive an invocation. See docs/fused-inference.md.
"""

import contextvars
import functools
import importlib
import logging
import os
import threading
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)
_PAIR = contextvars.ContextVar("boltz_fused_pair", default=None)
_DENOISER = contextvars.ContextVar("boltz_fused_denoiser", default=None)
_LOCK = threading.RLock()
_LAST = {}
_LOGGED = set()
_LOG_LOCK = threading.Lock()
_PAIR_ATTRS = (
    "_pairfuse_elig",
    "_pairfuse_w",
    "_pairfuse_core_w",
    "_opt_pairblock_W",
    "_opt_transition_W",
    "_opt_transition_T",
)


def components():
    value = os.environ.get("BOLTZ_FUSED_INFERENCE", "off").strip().lower()
    modes = {
        "off": (),
        "false": (),
        "0": (),
        "dit": ("dit",),
        "atom": ("atom",),
        "denoiser": ("dit", "atom"),
        "pair": ("pair",),
        "all": ("pair", "dit", "atom"),
        "on": ("pair", "dit", "atom"),
        "true": ("pair", "dit", "atom"),
        "1": ("pair", "dit", "atom"),
    }
    if value not in modes:
        raise ValueError(
            "BOLTZ_FUSED_INFERENCE must be off, dit, atom, denoiser, pair, or all"
        )
    return modes[value]


def _runtime_supported():
    version = torch.__version__.split("+")[0].split(".")
    return tuple(int(x) for x in version[:2]) >= (2, 12)


def _notice(component, reason=None, *, stack=False):
    """Log each requested component's refusal or first actual use once."""
    key = (component, reason, stack)
    with _LOG_LOCK:
        if key in _LOGGED:
            return
        _LOGGED.add(key)
    if reason is None:
        logger.info("Fused inference %s: using fused kernels", component)
    elif stack and (
        reason in ("below_min_tokens", "autocast_off", "training")
        or reason.startswith(("c:", "no_cell:"))
    ):
        logger.info("Fused inference pair: stack uses original path (%s)", reason)
    else:
        logger.warning(
            "Fused inference %s declined: %s; using original path", component, reason
        )


def _decline(parts, reason):
    for part in parts:
        _notice(part, reason)


def _unsafe_reason(module, tensor):
    if torch.compiler.is_compiling():
        return "compiling"
    if module.training:
        return "training"
    if torch.is_grad_enabled():
        return "gradients_enabled"
    if not tensor.is_cuda:
        return "not_cuda"
    if not _runtime_supported():
        return "requires_torch_2.12"
    if torch.get_default_dtype() != torch.float32:
        return "default_dtype_not_fp32"
    if tensor.device != torch.device("cuda", torch.cuda.current_device()):
        return "noncurrent_cuda_device"
    for child in module.modules():
        if child.training:
            return "training_submodule"
        if child._forward_hooks or child._forward_pre_hooks:
            return "forward_hooks"
        if hasattr(child, "_orig_mod"):
            return "compiled_module"
        if "forward" in child.__dict__:
            return "custom_forward"
        if not type(child).__module__.startswith(
            ("torch.", "boltz.", "einops.layers.", "torchmetrics.")
        ):
            return "custom_module"
        if not getattr(type(child).forward, "__module__", "").startswith(
            ("torch.", "boltz.", "einops.layers.", "torchmetrics.")
        ):
            return "custom_forward"
    return None


def _safe(module, tensor):
    return _unsafe_reason(module, tensor) is None


def _versioned_weights(module):
    try:
        return all(p._version >= 0 for p in module.parameters())
    except RuntimeError:
        return False


@dataclass
class PairScope:
    backend: object
    projections: dict = field(default_factory=dict)
    calls: int = 0
    fallbacks: dict = field(default_factory=dict)


def _load_kernels(name):
    return importlib.import_module(f"{__package__}._anthropic.{name}")


def forward_scope(function):
    @functools.wraps(function)
    def scoped(module, feats, *args, **kwargs):
        if "pair" not in components():
            return function(module, feats, *args, **kwargs)
        tensor = feats["token_pad_mask"]
        reason = None
        if not _safe(module, tensor):
            reason = _unsafe_reason(module, tensor) or "unsafe_context"
        elif (
            not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") != torch.bfloat16
        ):
            reason = "requires_cuda_bf16_autocast"
        elif not _versioned_weights(module):
            reason = "unversioned_weights"
        elif not _LOCK.acquire(blocking=False):
            reason = "lock_contention"
        if reason is not None:
            _notice("pair", reason)
            return function(module, feats, *args, **kwargs)
        backend = scope = token = None
        saved = []
        try:
            backend = _load_kernels("pair")
            saved = [
                (m, {n: m.__dict__[n] for n in _PAIR_ATTRS if n in m.__dict__})
                for m in module.modules()
            ]
            residency, picks = backend.parse_word(
                "bf16,trimul=core.fast,triatt=core.fast,transition=core.fast",
                environ={},
            )
            backend._STATE.update(residency=residency, picks=picks, handed_triatt={})
            scope = PairScope(backend)
            token = _PAIR.set(scope)
            return function(module, feats, *args, **kwargs)
        finally:
            try:
                if token is not None:
                    _PAIR.reset(token)
                if scope is not None:
                    _LAST["pair"] = {
                        "served_stacks": scope.calls,
                        "fallbacks": scope.fallbacks,
                    }
                    scope.projections.clear()
                if backend is not None:
                    backend._MASK_PRED.clear()
                    backend._MASK_GIVEN["value"] = None
                    backend._CORE_TRIMUL_CACHE.clear()
                for m, old in saved:
                    for name in _PAIR_ATTRS:
                        m.__dict__.pop(name, None)
                    m.__dict__.update(old)
                from . import flashpairformer

                flashpairformer.clear_weight_caches()
            finally:
                _LOCK.release()

    return scoped


def pair_stack(
    module, z, pair_mask, use_kernels, s=None, mask=None, use_flash_attn=False
):
    if torch.compiler.is_compiling():
        return None
    scope = _PAIR.get()
    if scope is None or module.training or torch.is_grad_enabled():
        return None
    layers = getattr(module, "layers", (module,))
    if not len(layers):
        return None
    backend = scope.backend
    reason = backend._stack_word(layers[0], z, use_kernels, module.training)
    if reason is not None:
        scope.fallbacks[reason] = scope.fallbacks.get(reason, 0) + 1
        _notice("pair", reason, stack=True)
        return None
    zr = backend._enter(z)
    ctx = backend._make_ctx(zr, pair_mask)
    for layer in layers:
        zr = backend._pair_sublayers(layer, zr, ctx)
        if s is not None:
            # Use the actual fused pair-bias projection without replacing a
            # registered model module; preserve the fork's SDPA choice.
            attention = layer.attention
            projection = scope.projections.get(id(attention))
            if projection is None:
                projection = backend.FusedProjZ(attention.proj_z)
                scope.projections[id(attention)] = projection
            bias = projection(zr)
            if use_flash_attn:
                normed = layer.pre_norm_s(s)
                s = s + attention(
                    s=normed,
                    z=zr,
                    mask=mask,
                    k_in=normed,
                    use_flash_attn=True,
                    precomputed_bias=bias.to(zr.dtype),
                )
                s = layer.s_post_norm(s + layer.transition_s(s))
            else:
                with torch.autocast(s.device.type, enabled=False):
                    normed = layer.pre_norm_s(s.float())
                    s = s.float() + attention(
                        s=normed,
                        z=zr,
                        mask=mask.float(),
                        k_in=normed,
                        precomputed_bias=bias.float(),
                    )
                    s = layer.s_post_norm(s + layer.transition_s(s))
    scope.calls += 1
    _notice("pair")
    out = backend._exit(zr, z)
    return out if s is None else (s, out)


@dataclass
class DenoiserScope:
    net: object
    atom: object = None
    dit: object = None
    atom_state: object = None
    calls: dict = field(default_factory=lambda: {"encoder": 0, "decoder": 0, "dit": 0})


def _geometry_unchecked(net, cond):
    if not isinstance(cond, dict) or "token_trans_bias" not in cond:
        return False
    bias = cond["token_trans_bias"]
    if bias.ndim != 4 or bias.shape[0] != 1:
        return False
    tt = net.token_transformer
    if (
        not len(tt.layers)
        or tt.layers[0].adaln.a_norm.normalized_shape != (768,)
        or tt.layers[0].pair_bias_attn.num_heads != 16
        or tt.layers[0].pair_bias_attn.head_dim != 48
    ):
        return False
    enc, dec = net.atom_attention_encoder, net.atom_attention_decoder
    return (
        enc.structure_prediction
        and enc.r_to_q_trans.weight.shape == (128, 3)
        and len(enc.atom_encoder.diffusion_transformer.layers) == 3
        and len(dec.atom_decoder.diffusion_transformer.layers) == 3
        and isinstance(dec.atom_feat_to_atom_pos_update, torch.nn.Sequential)
        and enc.atom_encoder.attn_window_queries == 32
        and enc.atom_encoder.attn_window_keys == 128
        and all(
            layer.pair_bias_attn.num_heads == 4
            and layer.pair_bias_attn.head_dim == 32
            and isinstance(layer.post_lnorm, torch.nn.Identity)
            for layer in list(enc.atom_encoder.diffusion_transformer.layers)
            + list(dec.atom_decoder.diffusion_transformer.layers)
        )
    )


def _geometry(net, cond):
    try:
        return _geometry_unchecked(net, cond)
    except (AttributeError, TypeError, IndexError):
        return False


def _packing_estimate(net, atom_mask, parts, chunk, n):
    estimate = 1024**3
    if "dit" in parts:
        estimate += 2 * len(net.token_transformer.layers) * 16 * n * n
    return estimate + chunk * (n * 768 * 24 * 16 + atom_mask.shape[1] * 128 * 64)


def _memory_headroom(device):
    free, _ = torch.cuda.mem_get_info(device)
    reclaimable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(
        device
    )
    return free + max(0, reclaimable)


def sampling_scope(function):
    @functools.wraps(function)
    def scoped(module, atom_mask, *args, **kwargs):
        requested = components()
        parts = tuple(x for x in requested if x in ("dit", "atom"))
        if not parts:
            return function(module, atom_mask, *args, **kwargs)
        if not _safe(module, atom_mask):
            _decline(parts, _unsafe_reason(module, atom_mask) or "unsafe_context")
            return function(module, atom_mask, *args, **kwargs)
        from boltz.model.modules.diffusionv2 import DiffusionModule
        from . import denoiser_inference

        cond = kwargs.get("diffusion_conditioning")
        net = module.score_model
        reason = None
        if type(net) is not DiffusionModule:
            reason = "custom_score_model"
        elif torch.is_autocast_enabled("cuda"):
            reason = "requires_autocast_off"
        elif torch.get_float32_matmul_precision() != "highest":
            reason = "requires_highest_matmul_precision"
        elif denoiser_inference.cache_enabled():
            reason = "denoiser_cache_enabled"
        elif not _versioned_weights(net):
            reason = "unversioned_weights"
        elif not _geometry(net, cond):
            reason = "unsupported_geometry"
        elif not _LOCK.acquire(blocking=False):
            reason = "lock_contention"
        if reason is not None:
            _decline(parts, reason)
            return function(module, atom_mask, *args, **kwargs)
        scope = DenoiserScope(net)
        attrs = []
        token = None
        try:
            n = cond["token_trans_bias"].shape[1]
            steering = kwargs.get("steering_args") or {}
            count = kwargs.get("multiplicity", 1)
            if steering.get("fk_steering"):
                count *= steering.get("num_particles", 1)
            cap = kwargs.get("max_parallel_samples") or count
            chunk = min(count, cap)
            # Conservative admission before any pack; OOM during computation
            # still propagates. No replay or retry of a scientific attempt.
            estimate = _packing_estimate(net, atom_mask, parts, chunk, n)
            if estimate + 2 * 1024**3 > _memory_headroom(atom_mask.device):
                _decline(parts, "insufficient_packing_headroom")
                return function(module, atom_mask, *args, **kwargs)
            if "atom" in parts:
                assignment = kwargs["feats"]["atom_to_token"]
                onehot = (((assignment == 0) | (assignment == 1)).all()) & (
                    assignment.sum(-1) <= 1
                ).all()
                if not bool(onehot):
                    _decline(parts, "atom_to_token_not_onehot_or_empty")
                    return function(module, atom_mask, *args, **kwargs)
            if "dit" in parts:
                dit = _load_kernels("dit")
                scope.dit = dit
                dit.configure(["bf16", "weights=sample"])
                dit.prepare(net, kwargs["feats"], cond, chunk)
            if "atom" in parts:
                atom = _load_kernels("atom")
                atom.new_epoch()
                scope.atom = atom
                scope.atom_state = atom.State(module, ("fused",), "bf16")
                for m in (net.atom_attention_encoder, net.atom_attention_decoder):
                    old = {
                        k: m.__dict__[k]
                        for k in ("_boltz_atom_state", "_boltz_atom_net")
                        if k in m.__dict__
                    }
                    attrs.append((m, old))
                    m.__dict__.update(
                        _boltz_atom_state=scope.atom_state, _boltz_atom_net=net
                    )
            token = _DENOISER.set(scope)
            return function(module, atom_mask, *args, **kwargs)
        finally:
            if token is not None:
                _DENOISER.reset(token)
            _LAST["denoiser"] = dict(scope.calls)
            if scope.dit is not None:
                scope.dit.release()
            if scope.atom_state is not None:
                scope.atom_state.release()
                scope.atom_state.w = None
            for m, old in attrs:
                m.__dict__.pop("_boltz_atom_state", None)
                m.__dict__.pop("_boltz_atom_net", None)
                m.__dict__.update(old)
            _LOCK.release()

    return scoped


def atom_encoder(
    module, feats, q, c, bias, to_keys, r, multiplicity, conditioning_cache
):
    if torch.compiler.is_compiling():
        return None
    scope = _DENOISER.get()
    if (
        scope is None
        or scope.atom is None
        or torch.is_grad_enabled()
        or conditioning_cache is not None
    ):
        return None
    if module is not scope.net.atom_attention_encoder:
        return None
    result = scope.atom._encoder_fused(
        module, feats, q, c, bias, to_keys, r=r, multiplicity=multiplicity
    )
    scope.calls["encoder"] += 1
    _notice("atom")
    return result


def atom_decoder(
    module, a, q, c, bias, feats, to_keys, multiplicity, conditioning_cache
):
    if torch.compiler.is_compiling():
        return None
    scope = _DENOISER.get()
    if (
        scope is None
        or scope.atom is None
        or torch.is_grad_enabled()
        or conditioning_cache is not None
    ):
        return None
    if module is not scope.net.atom_attention_decoder:
        return None
    result = scope.atom._decoder_fused(
        module, a, q, c, bias, feats, to_keys, multiplicity=multiplicity
    )
    scope.calls["decoder"] += 1
    _notice("atom")
    return result


def token_transformer(
    module, a, s, bias, mask, to_keys, multiplicity, conditioning_cache
):
    if torch.compiler.is_compiling():
        return None
    scope = _DENOISER.get()
    if (
        scope is None
        or scope.dit is None
        or torch.is_grad_enabled()
        or conditioning_cache is not None
    ):
        return None
    if module is not scope.net.token_transformer:
        return None
    if to_keys is not None or bias is None:
        _notice("dit", "not_token_attention")
        return None
    # Validate the functional backend contract here so unsupported inputs use
    # the original caller path, without invoking or re-entering class forward.
    state = scope.dit._STATE
    packed = state["st"] if state["tt"] is module else None
    if (
        not scope.dit._CFG["on"]
        or not module.pair_bias_attn
        or a.ndim != 3
        or packed is None
        or packed.pack is None
        or packed.pack["N"] != a.shape[1]
        or packed.D != a.shape[-1]
    ):
        _notice("dit", "missing_or_incompatible_pack")
        return None
    result = scope.dit.fused_forward(
        module, a, s, bias=bias, mask=mask, to_keys=to_keys, multiplicity=multiplicity
    )
    scope.calls["dit"] += 1
    _notice("dit")
    return result


def report():
    return dict(_LAST)
