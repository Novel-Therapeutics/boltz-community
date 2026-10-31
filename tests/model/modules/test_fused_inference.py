"""Admission, cleanup and actual fused inference; synthetic inputs only."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from boltz.model.layers import fused_inference as fused


@pytest.mark.parametrize(
    "value,expected",
    [
        ("off", ()),
        ("false", ()),
        ("0", ()),
        ("dit", ("dit",)),
        ("atom", ("atom",)),
        ("denoiser", ("dit", "atom")),
        ("pair", ("pair",)),
        ("all", ("pair", "dit", "atom")),
        ("true", ("pair", "dit", "atom")),
    ],
)
def test_switch(monkeypatch, value, expected):
    monkeypatch.setenv("BOLTZ_FUSED_INFERENCE", value)
    assert fused.components() == expected


def test_default_and_bad_switch(monkeypatch):
    monkeypatch.delenv("BOLTZ_FUSED_INFERENCE", raising=False)
    assert fused.components() == ()
    monkeypatch.setenv("BOLTZ_FUSED_INFERENCE", "typo")
    with pytest.raises(ValueError, match="BOLTZ_FUSED_INFERENCE"):
        fused.components()


def test_off_and_cpu_admission_preserve_original_rng(monkeypatch):
    module = torch.nn.Linear(2, 2).eval()

    @fused.forward_scope
    def forward(self, feats):
        return torch.rand(3)

    for setting in ["off", "all"]:
        monkeypatch.setenv("BOLTZ_FUSED_INFERENCE", setting)
        initial = torch.get_rng_state()
        expected = torch.rand(3)
        final = torch.get_rng_state()
        torch.set_rng_state(initial)
        with torch.no_grad():
            actual = forward(module, {"token_pad_mask": torch.ones(1, 2)})
        assert torch.equal(actual, expected) and torch.equal(
            torch.get_rng_state(), final
        )
    assert fused._PAIR.get() is None


def _lock_free():
    def check():
        ok = fused._LOCK.acquire(blocking=False)
        if ok:
            fused._LOCK.release()
        return ok

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(check).result()


@pytest.mark.parametrize("fail_setup", [False, True])
def test_pair_scope_restores_attrs_context_and_lock_on_error(monkeypatch, fail_setup):
    from boltz.model.layers import flashpairformer

    module = torch.nn.Linear(2, 2).eval()
    prior = object()
    module._opt_transition_W = prior
    backend = SimpleNamespace(
        _STATE={}, _MASK_PRED=[], _MASK_GIVEN={"value": None}, _CORE_TRIMUL_CACHE={}
    )

    def parse(word):
        if fail_setup:
            raise RuntimeError("setup failure")
        return "bf16", {}

    backend.parse_word = parse
    monkeypatch.setattr(fused, "_safe", lambda m, t: True)
    monkeypatch.setattr(fused, "_load_kernels", lambda name: backend)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *a: True)
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda *a: torch.bfloat16)
    monkeypatch.setattr(flashpairformer, "clear_weight_caches", lambda: None)
    monkeypatch.setenv("BOLTZ_FUSED_INFERENCE", "pair")

    @fused.forward_scope
    def forward(self, feats):
        self._opt_transition_W = object()
        raise RuntimeError("prediction failure")

    with pytest.raises(RuntimeError, match="failure"):
        forward(module, {"token_pad_mask": torch.ones(1, 2)})
    assert module._opt_transition_W is prior
    assert fused._PAIR.get() is None and _lock_free()


def test_failed_dit_pack_releases_partial_state(monkeypatch):
    from .test_denoiser_inference import _case

    sampler, mask, kwargs = _case("cpu")
    released = []

    def bad_prepare(*args):
        raise RuntimeError("pack failure")

    backend = SimpleNamespace(
        install=lambda *a: None,
        prepare=bad_prepare,
        release=lambda: released.append(True),
    )
    monkeypatch.setenv("BOLTZ_FUSED_INFERENCE", "dit")
    monkeypatch.setenv("BOLTZ_DENOISER_CACHE", "off")
    monkeypatch.setattr(fused, "_safe", lambda m, t: True)
    monkeypatch.setattr(fused, "_geometry", lambda *a: True)
    monkeypatch.setattr(fused, "_load_kernels", lambda name: backend)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (1024**4, 1024**4))
    with torch.no_grad(), pytest.raises(RuntimeError, match="pack failure"):
        sampler.sample(mask, **kwargs)
    assert released == [True] and fused._DENOISER.get() is None and _lock_free()


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused._runtime_supported(),
    reason="CUDA and Torch2.12+",
)
@pytest.mark.parametrize("setting", ["dit", "denoiser"])
def test_real_sampler_fused_route_uneven_chunks_rng_and_cleanup(monkeypatch, setting):
    from functools import partial
    from boltz.model.modules.diffusionv2 import AtomDiffusion
    from boltz.model.modules.encodersv2 import get_indexing_matrix, single_to_keys

    device = "cuda"
    torch.manual_seed(73)
    model = (
        AtomDiffusion(
            score_model_args=dict(
                token_s=384,
                atom_s=128,
                sigma_data=1,
                dim_fourier=8,
                atoms_per_window_queries=32,
                atoms_per_window_keys=128,
                atom_encoder_depth=3,
                atom_encoder_heads=4,
                atom_decoder_depth=3,
                atom_decoder_heads=4,
                token_transformer_depth=2,
                token_transformer_heads=16,
                conditioning_transition_layers=1,
            ),
            num_sampling_steps=3,
            sigma_data=1,
            sigma_min=0.1,
            sigma_max=1,
            gamma_0=0.2,
            gamma_min=0.5,
        )
        .to(device)
        .eval()
    )
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.02)
    mask = torch.ones(1, 64, device=device)
    mask[:, -3:] = 0
    assignment = (
        torch.nn.functional.one_hot(torch.arange(64, device=device) % 3, 3).float()[
            None
        ]
        * mask[..., None]
    )
    kwargs = dict(
        s_inputs=torch.randn(1, 3, 384, device=device),
        s_trunk=torch.randn(1, 3, 384, device=device),
        feats={
            "ref_pos": torch.randn(1, 64, 3, device=device),
            "atom_to_token": assignment,
            "atom_pad_mask": mask,
            "token_pad_mask": torch.ones(1, 3, device=device),
        },
        diffusion_conditioning={
            "q": torch.randn(1, 64, 128, device=device),
            "c": torch.randn(1, 64, 128, device=device),
            "atom_enc_bias": torch.randn(1, 2, 32, 128, 12, device=device),
            "atom_dec_bias": torch.randn(1, 2, 32, 128, 12, device=device),
            "token_trans_bias": torch.randn(1, 3, 3, 32, device=device),
            "to_keys": partial(
                single_to_keys,
                indexing_matrix=get_indexing_matrix(2, 32, 128, device),
                W=32,
                H=128,
            ),
        },
    )
    monkeypatch.setenv("BOLTZ_DENOISER_CACHE", "off")
    monkeypatch.setenv("BOLTZ_FUSED_INFERENCE", "off")
    with torch.inference_mode():
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state()
        expected = model.sample(mask, multiplicity=3, max_parallel_samples=2, **kwargs)
        cpu_end = torch.get_rng_state()
        cuda_end = torch.cuda.get_rng_state()
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda)
        monkeypatch.setenv("BOLTZ_FUSED_INFERENCE", setting)
        actual = model.sample(mask, multiplicity=3, max_parallel_samples=2, **kwargs)
        assert torch.equal(cpu_end, torch.get_rng_state()) and torch.equal(
            cuda_end, torch.cuda.get_rng_state()
        )
    assert torch.isfinite(actual["sample_atom_coords"]).all()
    torch.testing.assert_close(
        actual["sample_atom_coords"],
        expected["sample_atom_coords"],
        rtol=0.03,
        atol=0.03,
    )
    assert fused.report()["denoiser"]["dit"] > 0
    if setting == "denoiser":
        assert (
            fused.report()["denoiser"]["encoder"] > 0
            and fused.report()["denoiser"]["decoder"] > 0
        )
    assert fused._DENOISER.get() is None and _lock_free()
    assert "_boltz_atom_state" not in model.score_model.atom_attention_encoder.__dict__
    from boltz.model.layers._anthropic import dit

    assert dit._STATE["st"] is None and dit._STATE["tt"] is None


@pytest.mark.parametrize("sdpa", [False, True])
def test_precomputed_pair_bias_matches_original_projection(sdpa):
    from boltz.model.layers.attentionv2 import AttentionPairBias

    layer = AttentionPairBias(c_s=16, c_z=8, num_heads=4).eval()
    s = torch.randn(1, 5, 16)
    z = torch.randn(1, 5, 5, 8)
    mask = torch.ones(1, 5)
    with torch.no_grad():
        expected = layer(s, z, mask, s, use_flash_attn=sdpa)
        actual = layer(
            s, z, mask, s, use_flash_attn=sdpa, precomputed_bias=layer.proj_z(z)
        )
    assert torch.equal(actual, expected)


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused._runtime_supported(),
    reason="CUDA and Torch2.12+",
)
@pytest.mark.parametrize("tokens", [128, 846, 1380])
def test_real_pair_kernels_preserve_rng_input_and_registered_modules(
    monkeypatch, tokens
):
    from boltz.model.layers.pairformer import PairformerModule

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pair = PairformerModule(
                token_s=32, token_z=128, num_blocks=1, num_heads=4
            )

        @fused.forward_scope
        def forward(self, feats, s, z, mask, pair_mask):
            return self.pair(
                s, z, mask, pair_mask, use_kernels=True, use_flash_attn=True
            )

    original_safe = fused._safe
    monkeypatch.setattr(fused, "_safe", lambda m, t: original_safe(m.pair, t))
    torch.manual_seed(841 + tokens)
    model = Model().cuda().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.02)
        for m in model.modules():
            if isinstance(m, torch.nn.LayerNorm):
                if m.weight is not None:
                    m.weight.fill_(1)
                if m.bias is not None:
                    m.bias.zero_()
    s = torch.randn(1, tokens, 32, device="cuda")
    z = torch.randn(1, tokens, tokens, 128, device="cuda")
    original_z = z.clone()
    mask = torch.ones(1, tokens, device="cuda")
    mask[:, -5:] = 0
    pm = mask[:, :, None] * mask[:, None, :]
    feats = {"token_pad_mask": mask}
    module_ids = {n: id(m) for n, m in model.named_modules()}
    monkeypatch.setenv("BOLTZ_FUSED_INFERENCE", "off")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state()
        expected = model(feats, s, z, mask, pm)
        cpu_end = torch.get_rng_state()
        cuda_end = torch.cuda.get_rng_state()
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda)
        monkeypatch.setenv("BOLTZ_FUSED_INFERENCE", "pair")
        actual = model(feats, s, z, mask, pm)
        assert torch.equal(cpu_end, torch.get_rng_state()) and torch.equal(
            cuda_end, torch.cuda.get_rng_state()
        )
    assert not torch.equal(expected[1], z), "Nonzero pair update required"
    for a, b in zip(actual, expected):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=0.03, atol=0.03)
    assert torch.equal(original_z, z)
    assert module_ids == {n: id(m) for n, m in model.named_modules()}
    assert fused.report()["pair"]["served_stacks"] == 1
    from boltz.model.layers._anthropic import pair

    cell = pair.STATS["core_cells"][f"trimul@{tokens}"]
    assert f"triatt@{tokens}" in pair.STATS["core_cells"]
    if torch.cuda.get_device_capability() == (12, 0) and tokens == 1380:
        assert "tmk3_fast" in cell and "v4" in cell, cell
    assert not any(k in m.__dict__ for m in model.modules() for k in fused._PAIR_ATTRS)
    assert fused._PAIR.get() is None and _lock_free()


def test_custom_linear_is_refused_but_stock_rearrange_is_admitted(monkeypatch):
    from boltz.model.layers.attentionv2 import AttentionPairBias

    class Custom(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x) + 1

    fake_cuda = SimpleNamespace(is_cuda=True, device=torch.device("cuda", 0))
    monkeypatch.setattr(fused, "_runtime_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with torch.no_grad():
        assert not fused._safe(Custom(2, 2).eval(), fake_cuda)
        assert fused._safe(AttentionPairBias(16, 8, 4).eval(), fake_cuda)


def test_noncanonical_denoiser_head_layout_declines():
    from .test_denoiser_inference import _case

    sampler, _, kwargs = _case("cpu")
    assert not fused._geometry(sampler.score_model, kwargs["diffusion_conditioning"])


@pytest.mark.parametrize(
    "reason", ["off", "not_token", "no_state", "no_pack", "wrong_tokens", "wrong_width"]
)
def test_token_decline_does_not_reenter_class_forward(reason):
    module = SimpleNamespace(pair_bias_attn=reason != "not_token")
    packed = SimpleNamespace(
        pack=None
        if reason == "no_pack"
        else {"N": 4 if reason == "wrong_tokens" else 3},
        D=7 if reason == "wrong_width" else 8,
    )

    def reentry(*a, **kw):
        raise AssertionError("Upstream fallback must not re-enter our functional hook")

    backend = SimpleNamespace(
        _CFG={"on": reason != "off"},
        _STATE={"tt": module, "st": None if reason == "no_state" else packed},
        fused_forward=reentry,
    )
    scope = fused.DenoiserScope(SimpleNamespace(token_transformer=module), dit=backend)
    token = fused._DENOISER.set(scope)
    try:
        assert (
            fused.token_transformer(
                module, torch.ones(1, 3, 8), None, torch.ones(1), None, None, 1, None
            )
            is None
        )
        assert scope.calls["dit"] == 0
    finally:
        fused._DENOISER.reset(token)
