import pytest
from utils import assertDictEqual

from lightning_boost.modules.loss import TaskLoss
from lightning_boost.modules.metrics import TaskMetric
from lightning_boost.systems.utils import *
from tests.dummy.modules.loss import BCELoss
from tests.dummy.modules.metrics import BinaryAccuracy


def test_dasherize_class_name():
    class AwesomeClass:
        pass

    awesome_instance = AwesomeClass()

    assert dasherize_class_name(awesome_instance) == 'awesome-class'


def test_get_loss_dict():
    loss = TaskLoss(BCELoss())
    assertDictEqual(get_loss_dict(loss), {'base-task': {'bce-loss': loss}})

    loss1 = TaskLoss(BCELoss(), task='task-1')
    loss2 = TaskLoss(BCELoss(), task='task-2')
    assertDictEqual(
        get_loss_dict([loss1, loss2]),
        {'task-1': {'bce-loss': loss1}, 'task-2': {'bce-loss': loss2}},
    )


def test_get_metrics_dict():
    metric = TaskMetric(BinaryAccuracy())
    assertDictEqual(get_metrics_dict(metric), {'base-task': {'binary-accuracy': metric}})

    metric1 = TaskMetric(BinaryAccuracy(), task='task-1')
    metric2 = TaskMetric(BinaryAccuracy(), task='task-2')
    assertDictEqual(
        get_metrics_dict([metric1, metric2]),
        {'task-1': {'binary-accuracy': metric1}, 'task-2': {'binary-accuracy': metric2}},
    )


def test_get_models_dict():
    model = BaseModel()
    assertDictEqual(get_models_dict(model), {'base-model': model})

    model1 = BaseModel('model-1')
    model2 = BaseModel('model-2')
    assertDictEqual(get_models_dict([model1, model2]), {'model-1': model1, 'model-2': model2})
