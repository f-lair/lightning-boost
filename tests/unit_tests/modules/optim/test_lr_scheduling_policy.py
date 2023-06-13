import pytest
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR

from lightning_boost.modules.optim import LRSchedulingPolicy


def test_init():
    policy = LRSchedulingPolicy()
    assert policy.interval == 'epoch'
    assert policy.frequency == 1

    interval = 'step'
    frequency = 2
    policy = LRSchedulingPolicy(interval, frequency)
    assert policy.interval == interval
    assert policy.frequency == frequency


def test_bind_lr_scheduler(dummy_system):
    model = dummy_system.models['dummy-model']
    optimizer = SGD(model.parameters(), 1e-3)
    lr_scheduler = LinearLR(optimizer)

    policy = LRSchedulingPolicy()
    policy_dict = policy.bind_lr_scheduler(lr_scheduler)

    assert policy_dict['scheduler'] == lr_scheduler
    assert policy_dict['interval'] == policy.interval
    assert policy_dict['frequency'] == policy.frequency
