"""Layer test setup must preserve the caller's autograd mode."""

from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from . import test_outer_product_mean, test_triangle_attention


@pytest.mark.parametrize(
    "fixture_module",
    [test_outer_product_mean, test_triangle_attention],
    ids=["outer_product_mean", "triangle_attention"],
)
@pytest.mark.parametrize("grad_enabled", [True, False], ids=["grad", "no_grad"])
@pytest.mark.parametrize("setup_fails", [False, True], ids=["success", "failure"])
def test_layer_setup_preserves_grad_mode(
    fixture_module: ModuleType,
    grad_enabled: bool,
    setup_fails: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real fixtures, including failure during parameter initialization."""
    case = fixture_module.OuterProductMeanTest(methodName="test_chunk")

    if setup_fails:
        monkeypatch.setattr(
            fixture_module.nn.init,
            "normal_",
            Mock(side_effect=RuntimeError("injected initialization failure")),
        )

    # Set the incoming mode explicitly and restore it even if the assertion fails,
    # so this regression neither depends on test order nor leaks state itself.
    with torch.set_grad_enabled(grad_enabled):
        if setup_fails:
            with pytest.raises(RuntimeError, match="injected initialization failure"):
                case.setUp()
        else:
            case.setUp()
        assert torch.is_grad_enabled() is grad_enabled
