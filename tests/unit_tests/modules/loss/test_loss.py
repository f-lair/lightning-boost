import pytest
import torch
from torch.nn import MSELoss
from torch.testing import assert_close

from lightning_boost.modules.loss import TaskLoss


def test_init():
    loss_instance = MSELoss()

    loss = TaskLoss(loss_instance)
    assert loss.instance == loss_instance
    assert loss.task == 'base-task'
    assert loss.weight == 1.0

    task = 'test_task'
    weight = 0.5
    loss = TaskLoss(loss_instance, task, weight)
    assert loss.instance == loss_instance
    assert loss.task == task
    assert loss.weight == weight


def test_forward():
    loss_instance = MSELoss()
    loss = TaskLoss(loss_instance)
    y_hat = torch.randn(10)
    y = torch.randn(10)

    assert_close(loss(y_hat, y), loss_instance(y_hat, y))
