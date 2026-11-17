"""COM energies and atom gradients must use only the real chain atoms."""

import pytest
import torch

from boltz.model.potentials.potentials import SymmetricChainCOMPotential


def _features(
    chain_ids: tuple[int, int], padding: str = "unmapped"
) -> dict[str, torch.Tensor]:
    """Build unequal chains (three and four atoms) with two padding slots."""
    atom_to_token = torch.zeros(1, 9, 3)
    atom_to_token[0, :3, 0] = 1
    atom_to_token[0, 3:7, 1] = 1
    if padding == "existing_chain":
        atom_to_token[0, 7:, 0] = 1
    elif padding == "absent_chain":
        atom_to_token[0, 7:, 2] = 1
    return {
        "atom_to_token": atom_to_token,
        "asym_id": torch.tensor([[*chain_ids, 11]]),
        "symmetric_chain_index": torch.tensor([[chain_ids]]).transpose(1, 2),
        "atom_pad_mask": torch.tensor([[True] * 7 + [False] * 2]),
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_known_centers_and_chain_rule(dtype: torch.dtype) -> None:
    """Centers at x=2 and x=8 give energy 4 and per-atom forces 1/3, -1/4."""
    feats = _features((0, 1))
    coords = torch.zeros(1, 9, 3, dtype=dtype)
    coords[0, :7, 0] = torch.tensor([1, 2, 3, 5, 7, 9, 11], dtype=dtype)
    coords[0, 7:] = 1000
    pot = SymmetricChainCOMPotential()
    params = {"buffer": 10.0}

    energy = pot.compute(coords, feats, params)
    gradient = pot.compute_gradient(coords, feats, params)

    expected_gradient = torch.zeros_like(coords)
    expected_gradient[0, :3, 0] = 1 / 3
    expected_gradient[0, 3:7, 0] = -1 / 4
    assert energy.dtype == dtype
    assert gradient.dtype == dtype
    torch.testing.assert_close(energy, coords.new_tensor([4.0]))
    torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize("multiplicity", [1, 3])
@pytest.mark.parametrize("chain_ids", [(0, 1), (2, 5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("padding", ["unmapped", "existing_chain", "absent_chain"])
def test_gradient_matches_autograd(
    multiplicity: int,
    chain_ids: tuple[int, int],
    dtype: torch.dtype,
    padding: str,
) -> None:
    """Sample gradients match autograd with unequal sizes and arbitrary padding IDs."""
    feats = _features(chain_ids, padding)
    generator = torch.Generator().manual_seed(42)
    coords = torch.randn(multiplicity, 9, 3, dtype=dtype, generator=generator)
    coords[:, 3:7, 0] += 4
    # Interleave padding so neither reduction nor expansion can assume a valid prefix.
    order = torch.tensor([0, 7, 3, 1, 4, 8, 5, 2, 6])
    feats["atom_to_token"] = feats["atom_to_token"][:, order]
    feats["atom_pad_mask"] = feats["atom_pad_mask"][:, order]
    coords = coords[:, order].requires_grad_()
    pot = SymmetricChainCOMPotential()
    params = {"buffer": 10.0}

    energy = pot.compute(coords, feats, params)
    expected = torch.autograd.grad(energy.sum(), coords)[0]
    actual = pot.compute_gradient(coords.detach(), feats, params)

    assert energy.shape == (multiplicity,)
    assert torch.all(energy > 0)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[:, ~feats["atom_pad_mask"][0]]) == 0
    # A shared feature record must give each sample its own independent result.
    for sample in range(multiplicity):
        single_coords = coords.detach()[sample : sample + 1]
        torch.testing.assert_close(
            energy[sample : sample + 1], pot.compute(single_coords, feats, params)
        )
        torch.testing.assert_close(
            actual[sample : sample + 1],
            pot.compute_gradient(single_coords, feats, params),
        )


@pytest.mark.parametrize("chain_ids", [(0, 1), (2, 5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_translation_invariance(chain_ids: tuple[int, int], dtype: torch.dtype) -> None:
    """Translating unequal chains changes neither their COM energy nor gradients."""
    feats = _features(chain_ids)
    generator = torch.Generator().manual_seed(17)
    coords = torch.randn(3, 9, 3, dtype=dtype, generator=generator)
    coords[:, 3:7, 0] += 4
    translated = coords + coords.new_tensor([13.0, -7.0, 5.0])
    pot = SymmetricChainCOMPotential()
    params = {"buffer": 10.0}

    torch.testing.assert_close(
        pot.compute(coords, feats, params), pot.compute(translated, feats, params)
    )
    torch.testing.assert_close(
        pot.compute_gradient(coords, feats, params),
        pot.compute_gradient(translated, feats, params),
    )


def test_padding_coordinates_do_not_affect_energy_or_gradient() -> None:
    """Even nonfinite coordinates in unused padding must be ignored."""
    feats = _features((2, 5))
    generator = torch.Generator().manual_seed(23)
    coords = torch.randn(2, 9, 3, generator=generator)
    coords[:, 3:7, 0] += 4
    changed = coords.clone()
    changed[:, 7] = float("nan")
    changed[:, 8] = float("inf")
    pot = SymmetricChainCOMPotential()
    params = {"buffer": 10.0}

    torch.testing.assert_close(
        pot.compute(coords, feats, params), pot.compute(changed, feats, params)
    )
    torch.testing.assert_close(
        pot.compute_gradient(coords, feats, params),
        pot.compute_gradient(changed, feats, params),
    )


def test_no_pairs_returns_zero() -> None:
    """An empty symmetric-pair set still bypasses the COM reduction."""
    feats = _features((2, 5))
    feats["symmetric_chain_index"] = torch.empty((1, 2, 0), dtype=torch.long)
    coords = torch.zeros(3, 9, 3)
    pot = SymmetricChainCOMPotential()
    params = {"buffer": 10.0}

    torch.testing.assert_close(pot.compute(coords, feats, params), torch.zeros(3))
    torch.testing.assert_close(
        pot.compute_gradient(coords, feats, params), torch.zeros_like(coords)
    )
