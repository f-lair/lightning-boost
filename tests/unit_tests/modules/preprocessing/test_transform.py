import pytest
import torch
from torch.testing import assert_close

from lightning_boost.modules.preprocessing import BaseTransform, CompositeTransform
from tests.dummy.modules.preprocessing import DummyTransform


def test_init():
    transform = BaseTransform()
    with pytest.raises(NotImplementedError) as exc_info:
        transform({}, {})

    dummy_transform = DummyTransform()
    composite_transform = CompositeTransform([dummy_transform, dummy_transform])


def test_call():
    inputs = {'x': torch.ones((42,))}
    targets = {'y': torch.zeros(())}

    dummy_transform = DummyTransform()
    transformed_inputs, transformed_targets = dummy_transform(inputs, targets)
    assert_close(transformed_inputs['x'], torch.full_like(inputs['x'], 0.1))
    assert_close(transformed_targets['y'], torch.ones_like(targets['y']))

    composite_transform = CompositeTransform([dummy_transform, dummy_transform])
    transformed_inputs, transformed_targets = composite_transform(inputs, targets)
    assert_close(transformed_inputs['x'], torch.full_like(inputs['x'], 0.01))
    assert_close(transformed_targets['y'], torch.full_like(targets['y'], 2))
