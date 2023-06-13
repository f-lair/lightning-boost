import pytest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR
from torch.testing import assert_close
from utils import assertDictEqual

from lightning_boost.models import BaseModel
from lightning_boost.modules.loss import TaskLoss
from lightning_boost.modules.metrics import TaskMetric
from lightning_boost.modules.optim import LRSchedulingPolicy
from lightning_boost.systems import BaseSystem
from tests.dummy.modules.loss import BCELoss
from tests.dummy.modules.metrics import BinaryAccuracy


def test_init():
    model = BaseModel()
    loss = TaskLoss(BCELoss())
    metric = TaskMetric(BinaryAccuracy())
    optimizer = lambda params: SGD(params, lr=1e-3)
    lr_scheduler = lambda optim: LinearLR(optim)
    lr_scheduling_policy = LRSchedulingPolicy('step', 2)

    system = BaseSystem(
        model, loss, optimizer, lr_scheduler, lr_scheduling_policy, metric, metric, metric
    )

    assertDictEqual(system.models, {'base-model': model})
    assertDictEqual(system.loss_functions, {'base-task': {'bce-loss': loss}})
    assert system.optimizer == optimizer
    assert system.lr_scheduler == lr_scheduler
    assert system.lr_scheduling_policy == lr_scheduling_policy
    assertDictEqual(system.train_metrics, {'base-task': {'binary-accuracy': metric}})
    assertDictEqual(system.val_metrics, {'base-task': {'binary-accuracy': metric}})
    assertDictEqual(system.test_metrics, {'base-task': {'binary-accuracy': metric}})


def test_step(dummy_system):
    model = BaseModel()
    loss = TaskLoss(BCELoss())
    optimizer = lambda params: SGD(params, lr=1e-3)

    N, D = 4, 42
    inputs = {'x': torch.ones((N, D))}
    targets = {'y': torch.ones((N,))}

    system = BaseSystem(model, loss, optimizer)
    with pytest.raises(NotImplementedError) as exc_info:
        system.step(inputs, targets)

    y_hat = dummy_system.step(inputs, targets)
    assertDictEqual(y_hat, {'y': torch.full((N,), 0.5)})


def test_training_step(dummy_system):
    dummy_system.train()

    N, D = 4, 42
    inputs = {'x': torch.ones((N, D))}
    targets = {'y': torch.ones((N,))}
    loss_function = BCELoss()

    loss = dummy_system.training_step((inputs, targets), 0)
    expected_loss = loss_function(torch.full((N,), 0.5), targets['y'])
    assert_close(loss, expected_loss)

    targets = {'y': torch.ones((N,)), 'y-clone': torch.ones((N,))}
    with pytest.raises(AssertionError) as exc_info:
        loss = dummy_system.training_step((inputs, targets), 0)

    dummy_system.loss_functions['y'] = dummy_system.loss_functions['base-task']
    del dummy_system.loss_functions['base-task']
    dummy_system.loss_functions['y-clone'] = dummy_system.loss_functions['y']
    with pytest.raises(AssertionError) as exc_info:
        loss = dummy_system.training_step((inputs, targets), 0)

    dummy_system.train_metrics['y'] = dummy_system.train_metrics['base-task']
    del dummy_system.train_metrics['base-task']
    dummy_system.train_metrics['y-clone'] = dummy_system.train_metrics['y']
    loss = dummy_system.training_step((inputs, targets), 0)
    expected_loss *= 2
    assert_close(loss, expected_loss)


def test_validation_step(dummy_system):
    dummy_system.eval()

    N, D = 4, 42
    inputs = {'x': torch.ones((N, D))}
    targets = {'y': torch.ones((N,))}
    loss_function = BCELoss()

    loss = dummy_system.validation_step((inputs, targets), 0)
    expected_loss = loss_function(torch.full((N,), 0.5), targets['y'])
    assert_close(loss, expected_loss)

    targets = {'y': torch.ones((N,)), 'y-clone': torch.ones((N,))}
    with pytest.raises(AssertionError) as exc_info:
        loss = dummy_system.validation_step((inputs, targets), 0)

    dummy_system.loss_functions['y'] = dummy_system.loss_functions['base-task']
    del dummy_system.loss_functions['base-task']
    dummy_system.loss_functions['y-clone'] = dummy_system.loss_functions['y']
    with pytest.raises(AssertionError) as exc_info:
        loss = dummy_system.validation_step((inputs, targets), 0)

    dummy_system.val_metrics['y'] = dummy_system.val_metrics['base-task']
    del dummy_system.val_metrics['base-task']
    dummy_system.val_metrics['y-clone'] = dummy_system.val_metrics['y']
    loss = dummy_system.validation_step((inputs, targets), 0)
    expected_loss *= 2
    assert_close(loss, expected_loss)


def test_test_step(dummy_system):
    dummy_system.eval()

    N, D = 4, 42
    inputs = {'x': torch.ones((N, D))}
    targets = {'y': torch.ones((N,))}
    loss_function = BCELoss()

    loss = dummy_system.test_step((inputs, targets), 0)
    expected_loss = loss_function(torch.full((N,), 0.5), targets['y'])
    assert_close(loss, expected_loss)

    targets = {'y': torch.ones((N,)), 'y-clone': torch.ones((N,))}
    with pytest.raises(AssertionError) as exc_info:
        loss = dummy_system.test_step((inputs, targets), 0)

    dummy_system.loss_functions['y'] = dummy_system.loss_functions['base-task']
    del dummy_system.loss_functions['base-task']
    dummy_system.loss_functions['y-clone'] = dummy_system.loss_functions['y']
    with pytest.raises(AssertionError) as exc_info:
        loss = dummy_system.test_step((inputs, targets), 0)

    dummy_system.test_metrics['y'] = dummy_system.test_metrics['base-task']
    del dummy_system.test_metrics['base-task']
    dummy_system.test_metrics['y-clone'] = dummy_system.test_metrics['y']
    loss = dummy_system.test_step((inputs, targets), 0)
    expected_loss *= 2
    assert_close(loss, expected_loss)


def test_configure_optimizers(dummy_system):
    # dicts cannot be compared directly
    configuration = dummy_system.configure_optimizers()
    assert set(configuration.keys()) == {'optimizer'}
    assert isinstance(configuration['optimizer'], SGD)

    dummy_system.lr_scheduler = lambda optim: LinearLR(optim)
    lr_scheduling_policy = LRSchedulingPolicy('step', 2)
    dummy_system.lr_scheduling_policy = lr_scheduling_policy

    configuration = dummy_system.configure_optimizers()
    assert set(configuration.keys()) == {'optimizer', 'lr_scheduler'}
    assert isinstance(configuration['optimizer'], SGD)
    assert isinstance(configuration['lr_scheduler'], dict)
    assert set(configuration['lr_scheduler'].keys()) == {'scheduler', 'interval', 'frequency'}
    assert isinstance(configuration['lr_scheduler']['scheduler'], LinearLR)
    assert configuration['lr_scheduler']['interval'] == lr_scheduling_policy.interval
    assert configuration['lr_scheduler']['frequency'] == lr_scheduling_policy.frequency
