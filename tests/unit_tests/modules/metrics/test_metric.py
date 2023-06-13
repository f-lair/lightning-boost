import pytest
import torch
from torch.testing import assert_close
from torchmetrics.classification import BinaryAccuracy

from lightning_boost.modules.metrics import TaskMetric


def test_init():
    metric_instance = BinaryAccuracy()

    metric = TaskMetric(metric_instance)
    assert metric.instance == metric_instance
    assert metric.task == 'base-task'

    task = 'test_task'
    metric = TaskMetric(metric_instance, task)
    assert metric.instance == metric_instance
    assert metric.task == task


def test_metric():
    metric_instance = BinaryAccuracy()
    metric = TaskMetric(metric_instance.clone())
    y_hat = torch.rand(10)
    y = torch.randint(0, 2, (10,))

    metric(y_hat, y)
    metric_instance(y_hat, y)

    assert_close(metric.compute(), metric_instance.compute())
