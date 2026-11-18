"""Keep Blackwell adaptation scoped to the qualified sources and launch path."""

from types import SimpleNamespace

import pytest

from boltz.upstream import blackwell


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
