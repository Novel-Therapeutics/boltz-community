"""Synthetic graph tests; no pretrained checkpoint or scientific predictions."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from boltz.model.layers import denoiser_graph as graphs
from .test_denoiser_inference import _assert_state, _case, _restore, _state

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _observe(monkeypatch):
    seen = []
    base = graphs.DenoiserGraphs

    class Observed(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append(self)

    monkeypatch.setattr(graphs, "DenoiserGraphs", Observed)
    return seen


def test_default_off_cpu_and_compiled_models_bypass(monkeypatch):
    sampler, mask, kwargs = _case("cpu")
    monkeypatch.delenv("BOLTZ_DENOISER_GRAPH", raising=False)
    assert not graphs.graph_enabled()
    monkeypatch.setattr(
        graphs, "DenoiserGraphs", lambda *a: pytest.fail("created graph")
    )
    with torch.no_grad():
        sampler.sample(mask, **kwargs)
        monkeypatch.setenv("BOLTZ_DENOISER_GRAPH", "on")
        sampler.sample(mask, **kwargs)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert not graphs._eligible(sampler, mask)


def test_bad_setting_fails_before_admission(monkeypatch):
    monkeypatch.setenv("BOLTZ_DENOISER_GRAPH", "typo")
    monkeypatch.setattr(
        graphs, "_eligible", lambda *a: pytest.fail("admission reached")
    )
    with pytest.raises(ValueError, match="BOLTZ_DENOISER_GRAPH"):
        graphs.graph_sampling(lambda *a: None)(None, None)


def test_cleanup_synchronizes_graph_devices_and_cpu_inference_input_declines(
    monkeypatch,
):
    sampler, _, kwargs = _case("cpu")
    runner = graphs.DenoiserGraphs(sampler)
    with torch.inference_mode():
        args = {**kwargs, "unused_cpu_metadata": torch.ones(1)}
        assert runner._signature(args)[0] is None
    devices = []
    monkeypatch.setattr(torch.cuda, "synchronize", devices.append)
    runner.entries = {
        k: SimpleNamespace(coords=SimpleNamespace(device=torch.device("cuda", k)))
        for k in (1, 2)
    }
    runner.close()
    assert set(devices) == {torch.device("cuda", 1), torch.device("cuda", 2)}
    assert runner.closed and not runner.entries


def test_scope_cleanup_and_original_chunk_plumbing(monkeypatch):
    sampler, mask, kwargs = _case("cpu", batch=2)
    seen = []

    class EagerRunner:
        def __init__(self, module):
            self.module, self.shapes, self.closed = module, [], False
            self.captures = self.replays = self.eager_calls = self.invalidations = 0
            self.peak_pool_bytes = self.capture_seconds = 0
            self.reason = None
            seen.append(self)

        def run(self, coords, sigma, kwargs):
            self.shapes.append(coords.shape[0])
            return self.module.preconditioned_network_forward(
                coords, sigma, network_condition_kwargs=kwargs
            )

        def close(self):
            self.closed = True

    monkeypatch.setattr(graphs, "DenoiserGraphs", EagerRunner)
    monkeypatch.setattr(graphs, "_eligible", lambda *a: True)
    monkeypatch.setenv("BOLTZ_DENOISER_GRAPH", "on")
    with torch.inference_mode():
        sampler.sample(mask, multiplicity=11, max_parallel_samples=5, **kwargs)
    assert seen[0].closed and seen[0].shapes == [10, 10, 2] * 3
    monkeypatch.setattr(
        EagerRunner, "run", lambda *a: (_ for _ in ()).throw(ValueError("probe"))
    )
    with torch.inference_mode(), pytest.raises(ValueError, match="probe"):
        sampler.sample(mask, **kwargs)
    assert seen[-1].closed


@pytest.mark.parametrize("oom", [False, True])
def test_capture_errors_decline_once_but_oom_propagates(monkeypatch, oom):
    sampler, _, kwargs = _case("cpu")
    runner = graphs.DenoiserGraphs(sampler)
    coords = torch.randn(1, 8, 3)
    kwargs = {**kwargs, "multiplicity": 1}
    count = Counter()

    def capture(*args):
        count["capture"] += 1
        raise (
            torch.cuda.OutOfMemoryError("out of memory")
            if oom
            else RuntimeError("uncapturable")
        )

    monkeypatch.setattr(runner, "_capture", capture)
    with torch.no_grad():
        if oom:
            with pytest.raises(torch.cuda.OutOfMemoryError):
                runner.run(coords, 0.5, kwargs)
        else:
            a = runner.run(coords, 0.5, kwargs)
            b = runner.run(coords, 0.5, kwargs)
            assert torch.equal(a, b) and count["capture"] == 1
    runner.close()
    assert runner.closed and not runner.entries


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("cache", ["off", "on"])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("batch,multiplicity,cap", [(1, 1, 1), (2, 11, 5)])
def test_cuda_whole_sampler_parity_and_actual_replay(
    monkeypatch, cache, amp, batch, multiplicity, cap
):
    sampler, mask, kwargs = _case("cuda", batch)
    seen = _observe(monkeypatch)
    monkeypatch.setenv("BOLTZ_DENOISER_CACHE", cache)
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp),
    ):
        start = _state("cuda")
        monkeypatch.setenv("BOLTZ_DENOISER_GRAPH", "off")
        expected = sampler.sample(
            mask, multiplicity=multiplicity, max_parallel_samples=cap, **kwargs
        )
        end = _state("cuda")
        _restore(start)
        monkeypatch.setenv("BOLTZ_DENOISER_GRAPH", "on")
        actual = sampler.sample(
            mask, multiplicity=multiplicity, max_parallel_samples=cap, **kwargs
        )
        _assert_state(end, _state("cuda"))
    assert torch.equal(expected["sample_atom_coords"], actual["sample_atom_coords"])
    assert seen[0].captures == (1 if multiplicity == 1 else 2)
    assert seen[0].replays > 0 and seen[0].closed and not seen[0].entries
    assert 0 < seen[0].peak_pool_bytes <= seen[0].max_bytes


@pytest.mark.cuda
@CUDA
def test_cuda_guidance_alignment_and_particle_resampling(monkeypatch):
    from boltz.model.modules import diffusionv2

    class Potential:
        def compute_parameters(self, t):
            return dict(resampling_weight=1.0, guidance_weight=0.1, guidance_interval=1)

        def compute(self, coords, feats, params):
            return coords.square().sum(dim=(-1, -2))

        def compute_gradient(self, coords, feats, params):
            return coords * 0.01

    sampler, mask, kwargs = _case("cuda", batch=2)
    sampler.alignment_reverse_diff = True
    sampler.gamma_min = 0
    seen = _observe(monkeypatch)
    monkeypatch.setattr(diffusionv2, "get_potentials", lambda *a, **k: [Potential()])
    settings = dict(
        fk_steering=True,
        num_particles=2,
        fk_lambda=0.1,
        physical_guidance_update=True,
        contact_guidance_update=True,
    )
    with torch.inference_mode():
        start = _state("cuda")
        monkeypatch.setenv("BOLTZ_DENOISER_GRAPH", "off")
        expected = sampler.sample(
            mask,
            multiplicity=2,
            max_parallel_samples=3,
            steering_args=settings,
            **kwargs,
        )
        end = _state("cuda")
        _restore(start)
        monkeypatch.setenv("BOLTZ_DENOISER_GRAPH", "on")
        actual = sampler.sample(
            mask,
            multiplicity=2,
            max_parallel_samples=3,
            steering_args=settings,
            **kwargs,
        )
        _assert_state(end, _state("cuda"))
    assert torch.equal(expected["sample_atom_coords"], actual["sample_atom_coords"])
    assert seen[0].replays > 0 and seen[0].closed


@pytest.mark.cuda
@CUDA
def test_cuda_live_inference_inputs_weight_updates_and_budget(monkeypatch):
    sampler, _, kwargs = _case("cuda")
    runner = graphs.DenoiserGraphs(sampler)
    with torch.inference_mode():
        kwargs["s_trunk"] = kwargs["s_trunk"].clone()  # no version counter
        coords = torch.randn(1, 8, 3, device="cuda")
        args = {**kwargs, "multiplicity": 1}
        runner.run(coords, 0.5, args)
        kwargs["s_trunk"].add_(0.1)
        actual = runner.run(coords, 0.7, args).clone()
        expected = sampler.preconditioned_network_forward(coords, 0.7, args)
        assert torch.equal(actual, expected) and runner.replays == 1
        sampler.score_model.single_conditioner.single_embed.weight.add_(0.01)
        actual = runner.run(coords, 0.3, args)
        expected = sampler.preconditioned_network_forward(coords, 0.3, args)
        assert torch.equal(actual, expected) and runner.invalidations == 1
    runner.close()
    limited = graphs.DenoiserGraphs(sampler, max_bytes=1)
    with torch.inference_mode():
        actual = limited.run(coords, 0.4, args)
        expected = sampler.preconditioned_network_forward(coords, 0.4, args)
    assert torch.equal(actual, expected) and not limited.captures and limited.declined
    limited.close()
