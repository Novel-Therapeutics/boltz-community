"""Explicit precision guards must survive ambient CPU autocast."""

import torch
from torch.nn import functional

from boltz.model.layers.triangular_attention.primitives import Linear
from boltz.model.loss.bfactor import bfactor_loss_fn


def test_linear_explicit_precision_overrides_cpu_autocast() -> None:
    """An fp32 projection keeps fp32 outputs and gradients under bf16 autocast."""
    layer = Linear(8, 4, precision=torch.float32)
    x = torch.randn(2, 8, requires_grad=True)
    with torch.enable_grad():
        with torch.autocast("cpu", enabled=False):
            expected = functional.linear(x, layer.weight, layer.bias)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = layer(x)
        actual.sum().backward()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, layer.weight.detach().sum(0).expand_as(x))


def test_bfactor_loss_preserves_fp32_under_cpu_autocast(
    small_feats_dict: dict[str, torch.Tensor],
) -> None:
    """Loss values and gradients agree with the fp32 path inside bf16 autocast."""
    pred = torch.randn(1, 8, 10, requires_grad=True)
    with torch.enable_grad():
        with torch.autocast("cpu", enabled=False):
            expected = bfactor_loss_fn({"pbfactor": pred}, small_feats_dict)
        expected_grad = torch.autograd.grad(expected, pred)[0]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = bfactor_loss_fn({"pbfactor": pred}, small_feats_dict)
        actual_grad = torch.autograd.grad(actual, pred)[0]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
