import torch

from boltz.model.modules.confidence import (
    _concat_confidence_outputs as concat_confidence_v1,
)
from boltz.model.modules.confidencev2 import (
    _concat_confidence_outputs as concat_confidence_v2,
)


def _sample_outputs():
    return [
        {
            "ptm": torch.tensor([1.0]),
            "chains_pae": {
                0: torch.tensor([2.0]),
            },
            "pair_chains_pae": {
                0: {
                    0: torch.tensor([3.0]),
                    1: torch.tensor([4.0]),
                }
            },
            "pair_chains_iptm": {
                0: {
                    0: torch.tensor([5.0]),
                }
            },
        },
        {
            "ptm": torch.tensor([10.0]),
            "chains_pae": {
                0: torch.tensor([20.0]),
            },
            "pair_chains_pae": {
                0: {
                    0: torch.tensor([30.0]),
                    1: torch.tensor([40.0]),
                }
            },
            "pair_chains_iptm": {
                0: {
                    0: torch.tensor([50.0]),
                }
            },
        },
    ]


def test_concat_confidence_outputs_v1_handles_nested_dicts():
    result = concat_confidence_v1(_sample_outputs())

    assert torch.equal(result["ptm"], torch.tensor([1.0, 10.0]))
    assert torch.equal(result["chains_pae"][0], torch.tensor([2.0, 20.0]))
    assert torch.equal(
        result["pair_chains_pae"][0][1], torch.tensor([4.0, 40.0])
    )
    assert torch.equal(
        result["pair_chains_iptm"][0][0], torch.tensor([5.0, 50.0])
    )


def test_concat_confidence_outputs_v2_handles_nested_dicts():
    result = concat_confidence_v2(_sample_outputs())

    assert torch.equal(result["ptm"], torch.tensor([1.0, 10.0]))
    assert torch.equal(result["chains_pae"][0], torch.tensor([2.0, 20.0]))
    assert torch.equal(
        result["pair_chains_pae"][0][1], torch.tensor([4.0, 40.0])
    )
    assert torch.equal(
        result["pair_chains_iptm"][0][0], torch.tensor([5.0, 50.0])
    )
