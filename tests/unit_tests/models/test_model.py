import pytest
import torch

from lightning_boost.models import BaseModel


def test_init():
    model = BaseModel()
    assert model.name is None

    name = "TestName"
    model = BaseModel(name)
    assert model.name == name

    with pytest.raises(NotImplementedError) as exc_info:
        model()


def test_forward(dummy_system):
    model = dummy_system.models['dummy-model']
    x = torch.ones((1, 42))
    y = model(x)
    assert y == 0.5
