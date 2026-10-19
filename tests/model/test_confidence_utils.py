import math

import pytest
import torch

from boltz.model.layers.confidence_utils import (
    compute_frame_pred,
    compute_pae_summaries,
    compute_ptms,
    compute_unweighted_pae_summaries,
)


def test_compute_pae_summaries_are_contact_weighted_and_nan_when_undefined():
    pae = torch.tensor(
        [
            [
                [0.0, 2.0, 4.0, 20.0],
                [6.0, 0.0, 8.0, 22.0],
                [10.0, 12.0, 0.0, 24.0],
                [26.0, 28.0, 30.0, 0.0],
            ]
        ]
    )
    asym_id = torch.tensor([[0, 0, 1, 1]])
    token_pad_mask = torch.tensor([[True, True, True, False]])

    pred_distogram_logits = torch.full((1, 4, 4, 64), -20.0)
    pred_distogram_logits[..., 20] = 20.0
    for i, j in [(0, 1), (1, 0), (0, 2), (2, 0)]:
        pred_distogram_logits[0, i, j, 20] = -20.0
        pred_distogram_logits[0, i, j, 0] = 20.0

    feats = {
        "asym_id": asym_id,
        "token_pad_mask": token_pad_mask,
    }

    complex_pae, complex_ipae, chains_pae, pair_chains_pae = compute_pae_summaries(
        pae, pred_distogram_logits, feats, multiplicity=1
    )

    assert torch.allclose(complex_pae, torch.tensor([5.5]))
    assert torch.allclose(complex_ipae, torch.tensor([7.0]))
    assert torch.allclose(chains_pae[0], torch.tensor([4.0]))
    assert math.isnan(chains_pae[1].item())
    assert torch.allclose(pair_chains_pae[0][0], torch.tensor([4.0]))
    assert torch.allclose(pair_chains_pae[0][1], torch.tensor([4.0]))
    assert torch.allclose(pair_chains_pae[1][0], torch.tensor([10.0]))
    assert math.isnan(pair_chains_pae[1][1].item())


# --- Unweighted PAE summaries -------------------------------------------------
# Hand-computed cases ported from upstream Boltz PR #707 by @AtharvaTilewale
# (https://github.com/jwohlwend/boltz/pull/707), which defines these values,
# so the *_unweighted fields stay comparable with upstream.

def _feats(asym_ids, pad_mask=None):
    asym_id = torch.tensor([asym_ids])
    if pad_mask is None:
        pad_mask = torch.ones_like(asym_id)
    else:
        pad_mask = torch.tensor([pad_mask])
    return {"asym_id": asym_id, "token_pad_mask": pad_mask}


def _block_pae(asym_ids, values):
    """PAE matrix whose value depends only on the (aligned, scored) chain pair."""
    n = len(asym_ids)
    pae = torch.zeros(1, n, n)
    for i in range(n):
        for j in range(n):
            pae[0, i, j] = values[(asym_ids[i], asym_ids[j])]
    return pae


def test_unweighted_two_chain_block_means():
    asym = [0, 0, 0, 1, 1]
    values = {(0, 0): 1.0, (0, 1): 8.0, (1, 0): 6.0, (1, 1): 2.0}
    complex_pae, complex_ipae, chains, pair = compute_unweighted_pae_summaries(
        _block_pae(asym, values), _feats(asym), 1
    )
    # Off-diagonal pairs: 3*2 intra chain 0, 2*1 intra chain 1, 6 + 6 inter.
    expected_complex = (6 * 1.0 + 2 * 2.0 + 6 * 8.0 + 6 * 6.0) / 20
    assert complex_pae.shape == (1,)
    assert torch.isclose(complex_pae[0], torch.tensor(expected_complex))
    assert torch.isclose(complex_ipae[0], torch.tensor((6 * 8.0 + 6 * 6.0) / 12))
    assert torch.isclose(pair[0][0][0], torch.tensor(1.0))
    assert torch.isclose(pair[0][1][0], torch.tensor(8.0))
    assert torch.isclose(pair[1][0][0], torch.tensor(6.0))
    assert torch.isclose(pair[1][1][0], torch.tensor(2.0))
    assert torch.equal(chains[0], pair[0][0])
    assert torch.equal(chains[1], pair[1][1])


def test_unweighted_diagonal_excluded():
    pae = torch.tensor([[[100.0, 3.0], [5.0, 100.0]]])
    complex_pae, _, chains, _ = compute_unweighted_pae_summaries(pae, _feats([0, 0]), 1)
    assert torch.isclose(complex_pae[0], torch.tensor(4.0))
    assert torch.isclose(chains[0][0], torch.tensor(4.0))


def test_unweighted_single_chain_interface_is_nan():
    pae = torch.full((1, 3, 3), 2.0)
    complex_pae, complex_ipae, chains, pair = compute_unweighted_pae_summaries(
        pae, _feats([0, 0, 0]), 1
    )
    assert torch.isclose(complex_pae[0], torch.tensor(2.0))
    assert torch.isnan(complex_ipae[0])
    assert list(pair) == [0]
    assert list(chains) == [0]


def test_unweighted_padding_and_frame_mask():
    asym = [0, 0, 1, 1]
    pad = [1, 1, 1, 0]  # last token is padding
    values = {(0, 0): 1.0, (0, 1): 4.0, (1, 0): 9.0, (1, 1): 2.0}
    pae = _block_pae(asym, values)
    feats = _feats(asym, pad)

    _, complex_ipae, _, pair = compute_unweighted_pae_summaries(pae, feats, 1)
    # Valid inter-chain pairs: (0,2), (1,2) -> 4.0 ; (2,0), (2,1) -> 9.0
    assert torch.isclose(complex_ipae[0], torch.tensor((2 * 4.0 + 2 * 9.0) / 4))
    # Chain 1 has a single valid token -> no intra-chain pair.
    assert torch.isnan(pair[1][1][0])

    # Masking the frame of token 2 removes every row aligned on chain 1.
    frame_mask = torch.tensor([[1, 1, 0, 1]])
    _, complex_ipae, _, pair = compute_unweighted_pae_summaries(
        pae, feats, 1, frame_mask=frame_mask
    )
    assert torch.isclose(complex_ipae[0], torch.tensor(4.0))
    assert torch.isnan(pair[1][0][0])
    assert torch.isclose(pair[0][1][0], torch.tensor(4.0))


def test_unweighted_multiplicity_batches_are_independent():
    pae = torch.stack(
        [
            torch.tensor([[0.0, 2.0], [4.0, 0.0]]),
            torch.tensor([[0.0, 10.0], [20.0, 0.0]]),
        ]
    )
    complex_pae, complex_ipae, _, pair = compute_unweighted_pae_summaries(
        pae, _feats([0, 1]), 2
    )
    assert complex_pae.shape == (2,)
    assert torch.allclose(complex_pae, torch.tensor([3.0, 15.0]))
    assert torch.allclose(complex_ipae, torch.tensor([3.0, 15.0]))
    assert torch.allclose(pair[0][1], torch.tensor([2.0, 10.0]))
    assert torch.allclose(pair[1][0], torch.tensor([4.0, 20.0]))


def test_unweighted_full_frame_mask_matches_plain_off_diagonal_mean():
    pae = torch.rand(1, 3, 3) * 30
    feats = _feats([0, 0, 1])
    a, _, _, _ = compute_unweighted_pae_summaries(pae, feats, 1)
    b, _, _, _ = compute_unweighted_pae_summaries(pae, feats, 1, frame_mask=torch.ones(1, 3))
    off_diag = pae[0][~torch.eye(3, dtype=torch.bool)]
    assert torch.isclose(a[0], off_diag.mean())
    assert torch.isclose(a[0], b[0])
    assert not math.isnan(a[0].item())


def test_contact_weighted_equals_unweighted_when_contact_probability_is_uniform():
    """The two families differ only in the contact weighting.

    With the same contact probability on every pair and no frame mask, the
    weights cancel and both must give identical values.
    """
    torch.manual_seed(0)
    asym = [0, 0, 1, 1, 2]
    pae = torch.rand(1, 5, 5) * 30
    feats = _feats(asym, [1, 1, 1, 1, 0])
    uniform_logits = torch.zeros(1, 5, 5, 64)  # equal bins -> equal contact prob

    weighted = compute_pae_summaries(pae, uniform_logits, feats, multiplicity=1)
    unweighted = compute_unweighted_pae_summaries(pae, feats, 1)

    assert torch.allclose(weighted[0], unweighted[0])
    assert torch.allclose(weighted[1], unweighted[1])
    assert set(weighted[3]) == set(unweighted[3])
    for c1 in weighted[3]:
        for c2 in weighted[3][c1]:
            assert torch.allclose(
                weighted[3][c1][c2], unweighted[3][c1][c2], equal_nan=True
            )


def test_contact_weighting_changes_the_values_when_contacts_vary():
    """Guard the converse: non-uniform contacts must move the weighted values."""
    asym = [0, 0, 1, 1]
    pae = torch.tensor(
        [[[0.0, 2.0, 4.0, 20.0], [6.0, 0.0, 8.0, 22.0],
          [10.0, 12.0, 0.0, 24.0], [26.0, 28.0, 30.0, 0.0]]]
    )
    logits = torch.full((1, 4, 4, 64), -20.0)
    logits[..., 20] = 20.0  # no contact
    logits[0, 0, 2, 20] = -20.0
    logits[0, 0, 2, 0] = 20.0  # only (0, 2) in contact
    weighted_ipae = compute_pae_summaries(pae, logits, _feats(asym), 1)[1]
    unweighted_ipae = compute_unweighted_pae_summaries(pae, _feats(asym), 1)[1]
    assert torch.isclose(weighted_ipae[0], torch.tensor(4.0), atol=1e-3)
    assert torch.isclose(unweighted_ipae[0], torch.tensor((4 + 20 + 8 + 22 + 10 + 12 + 26 + 28) / 8))


@pytest.mark.parametrize("seed", [0, 1])
def test_compute_ptms_accepts_precomputed_frame_mask(seed):
    """Passing the frame mask must give the same result as recomputing it."""
    torch.manual_seed(seed)
    n_tokens, n_atoms = 6, 18  # 3 atoms per token, two protein chains
    feats = {
        "asym_id": torch.tensor([[0, 0, 0, 1, 1, 1]]),
        "token_pad_mask": torch.ones(1, n_tokens),
        "atom_pad_mask": torch.ones(1, n_atoms),
        "mol_type": torch.zeros(1, n_tokens, dtype=torch.long),
        "atom_to_token": torch.repeat_interleave(torch.eye(n_tokens), 3, dim=0)[None],
        "frames_idx": torch.arange(n_atoms).reshape(1, n_tokens, 3),
    }
    coords = torch.randn(1, n_atoms, 3)
    logits = torch.randn(1, n_tokens, n_tokens, 64)

    _, frame_mask = compute_frame_pred(
        coords, feats["frames_idx"], feats, 1, inference=True
    )
    ref = compute_ptms(logits, coords, feats, 1)
    out = compute_ptms(logits, coords, feats, 1, frame_mask=frame_mask)
    for a, b in zip(ref[:4], out[:4]):
        assert torch.allclose(a, b)
    for c1 in ref[4]:
        for c2 in ref[4][c1]:
            assert torch.allclose(ref[4][c1][c2], out[4][c1][c2])
