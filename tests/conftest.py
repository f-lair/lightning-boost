import pytest
from torch.optim import SGD

from lightning_boost.modules.loss import TaskLoss
from lightning_boost.modules.metrics import TaskMetric
from tests.dummy.data.datamodules import DummyDatamodule
from tests.dummy.models import DummyModel
from tests.dummy.modules.loss import BCELoss
from tests.dummy.modules.metrics import BinaryAccuracy
from tests.dummy.systems import DummySystem


@pytest.fixture()
def dummy_datamodule():
    datamodule = DummyDatamodule()
    datamodule.setup('fit')
    datamodule.setup('test')

    return datamodule


@pytest.fixture()
def dummy_system():
    model = DummyModel()
    loss = TaskLoss(BCELoss())
    metric = TaskMetric(BinaryAccuracy())
    optimizer = lambda params: SGD(params, lr=1e-3)

    return DummySystem(
        models=model,
        loss=loss,
        optimizer=optimizer,
        train_metrics=metric,
        val_metrics=metric,
        test_metrics=metric,
    )
