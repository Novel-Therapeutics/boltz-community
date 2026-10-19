import pytest
import torch

from boltz.model.optim.scheduler import AlphaFoldLRScheduler


def _scheduler(**kwargs):
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    return optimizer, AlphaFoldLRScheduler(optimizer, **kwargs)


def test_zero_warmup_starts_at_max_lr_without_division_by_zero():
    """warmup_no_steps=0 used to divide by zero at step 0 (upstream PR #709)."""
    optimizer, scheduler = _scheduler(
        base_lr=0.0, max_lr=1e-3, warmup_no_steps=0, start_decay_after_n_steps=10
    )
    assert scheduler.get_last_lr() == [pytest.approx(1e-3)]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [pytest.approx(1e-3)]


def test_warmup_is_unchanged_when_enabled():
    optimizer, scheduler = _scheduler(
        base_lr=0.0, max_lr=1e-3, warmup_no_steps=4, start_decay_after_n_steps=10
    )
    lrs = [scheduler.get_last_lr()[0]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    assert lrs == [pytest.approx(x) for x in (0.0, 2.5e-4, 5e-4, 7.5e-4, 1e-3)]
