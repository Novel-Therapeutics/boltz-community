"""Keep Blackwell adaptation scoped to the qualified sources and launch path."""

from types import SimpleNamespace

import pytest

from boltz.upstream import blackwell


@pytest.mark.parametrize("batched", [False, True])
def test_stock_triangle_bias_replaces_inherited_projection(batched):
    import torch

    layer = torch.nn.Linear(128, 4, bias=False)
    module = SimpleNamespace(linear=layer)
    x = torch.randn((1, 5, 7, 128) if batched else (5, 7, 128))
    expected = layer(x).movedim(-1, -3).unsqueeze(-4)
    if not batched:
        expected = expected.unsqueeze(0)
    calls = []

    def core(*args):
        calls.append(args)
        return args[3]

    wrapped = blackwell.stock_triangle_bias_core(module, x, core)
    q, k, v, mask = (object() for _ in range(4))
    # Simulate the inherited projection's wrong values. They must never reach
    # attention, while the fused Q/K/V and mask/scale arguments pass through.
    wrong = expected.detach() + 1
    got = wrapped(q, k, v, wrong, mask, 0.125)
    assert torch.equal(got, expected)
    assert calls[0][:3] == (q, k, v)
    assert calls[0][4:] == (mask, 0.125)


def test_layernorm_requires_matching_runtime_and_source(tmp_path, monkeypatch):
    source = tmp_path / "__init__.py"
    source.write_bytes(b"qualified")
    package = SimpleNamespace(__file__=str(source), TORCH_PIN="2.12.0")
    import hashlib

    monkeypatch.setattr(
        blackwell,
        "EXACTLN_FILES",
        {source.name: hashlib.sha256(b"qualified").hexdigest()},
    )
    assert blackwell.admits_layernorm(package, (12, 0), "2.12.0+cu130")
    assert not blackwell.admits_layernorm(package, (9, 0), "2.12.0+cu130")
    assert not blackwell.admits_layernorm(package, (12, 0), "2.12.1+cu130")
    source.write_bytes(b"changed")
    assert not blackwell.admits_layernorm(package, (12, 0), "2.12.0+cu130")
    source.unlink()
    assert not blackwell.admits_layernorm(package, (12, 0), "2.12.0+cu130")


@pytest.mark.parametrize("cell", ["dit", "trans2x", "pwa2x"])
def test_exact_cell_admission_requires_probed_source_and_stack(tmp_path, monkeypatch, cell):
    import hashlib

    source = tmp_path / "cell.py"
    source.write_bytes(b"qualified")
    module = SimpleNamespace(__file__=str(source))
    monkeypatch.setattr(
        blackwell, "EXACT_CELL_FILES", {cell: hashlib.sha256(b"qualified").hexdigest()},
    )
    assert blackwell.admits_exact_cell(module, cell, (12, 0), "2.12.0+cu130", "13.0")
    assert not blackwell.admits_exact_cell(module, cell, (10, 0), "2.12.0+cu130", "13.0")
    assert not blackwell.admits_exact_cell(module, cell, (12, 0), "2.12.1+cu130", "13.0")
    assert not blackwell.admits_exact_cell(module, cell, (12, 0), "2.12.0+cu130", "13.1")
    source.write_bytes(b"changed")
    assert not blackwell.admits_exact_cell(module, cell, (12, 0), "2.12.0+cu130", "13.0")
    source.unlink()
    assert not blackwell.admits_exact_cell(module, cell, (12, 0), "2.12.0+cu130", "13.0")


def test_blackwell_opm_preserves_fusion_with_smaller_shared_tile():
    source = dict(BI=4, BJ=8, BK=64, BZ=32, num_stages=3, TMA=True, PM=1, FP=True)
    provider = SimpleNamespace(
        cfg_for=lambda *_: (dict(source), "default"),
        cfg_name=lambda _: "f1",
        CFG_VARIANTS={},
    )
    assert blackwell.configure_opm(provider, (9, 0)) is None
    assert not provider.CFG_VARIANTS
    assert blackwell.configure_opm(provider, (12, 0)) == blackwell.OPM_CONFIG
    assert provider.CFG_VARIANTS[blackwell.OPM_CONFIG] == {
        **source,
        "BJ": 4,
        "num_stages": 2,
    }
    assert source["BJ"] == 8


@pytest.mark.parametrize("origin", ["cc120", "override:custom"])
def test_provider_and_explicit_opm_choices_keep_precedence(origin):
    provider = SimpleNamespace(
        cfg_for=lambda *_: ({}, origin),
        cfg_name=lambda _: "f1",
        CFG_VARIANTS={},
    )
    assert blackwell.configure_opm(provider, (12, 0)) is None
    assert not provider.CFG_VARIANTS
