from pathlib import Path
from typing import Type

import pytest
import torch
from torch.testing import assert_close
from torch.utils.data import DataLoader, Dataset

from lightning_boost.data.datamodules import BaseDatamodule
from lightning_boost.data.datasets import BaseDataset
from lightning_boost.modules.preprocessing import BaseCollator, BaseTransform
from tests.dummy.data.datamodules import DummyDatamodule
from tests.dummy.data.datasets import DummyDataset


def test_init():
    with pytest.raises(NotImplementedError) as exc_info:
        datamodule = BaseDatamodule()

    datamodule = DummyDatamodule()

    class BaseDatamoduleA(BaseDatamodule):
        def get_collator(self, **kwargs) -> BaseCollator:
            return datamodule.get_collator()

    class BaseDatamoduleB(BaseDatamoduleA):
        def get_dataset_type(self, **kwargs) -> Type[BaseDataset]:
            return datamodule.get_dataset_type()

    class BaseDatamoduleC(BaseDatamoduleB):
        def get_transform(self, **kwargs) -> BaseTransform:
            return datamodule.get_transform()

    with pytest.raises(NotImplementedError) as exc_info:
        datamodule = BaseDatamoduleA()

    with pytest.raises(NotImplementedError) as exc_info:
        datamodule = BaseDatamoduleB()

    with pytest.raises(NotImplementedError) as exc_info:
        datamodule = BaseDatamoduleC()
        datamodule.setup('fit')

    with pytest.raises(AssertionError) as exc_info:
        datamodule = DummyDatamodule(batch_size=0)

    with pytest.raises(AssertionError) as exc_info:
        datamodule = DummyDatamodule(num_workers=-1)

    with pytest.raises(AssertionError) as exc_info:
        datamodule = DummyDatamodule(num_gpus=-1)

    with pytest.raises(AssertionError) as exc_info:
        datamodule = DummyDatamodule(val_ratio=-1.0)

    with pytest.raises(AssertionError) as exc_info:
        datamodule = DummyDatamodule(val_ratio=2.0)

    with pytest.raises(AssertionError) as exc_info:
        datamodule = DummyDatamodule(test_ratio=-1.0)

    with pytest.raises(AssertionError) as exc_info:
        datamodule = DummyDatamodule(test_ratio=2.0)

    with pytest.raises(AssertionError) as exc_info:
        datamodule = DummyDatamodule(val_ratio=0.6, test_ratio=0.6)


def test_prepare_data(dummy_datamodule):
    dummy_datamodule.prepare_data()

    data_path = Path(dummy_datamodule.data_dir)
    assert data_path.exists() and data_path.is_dir()


def test_instantiate_dataset(dummy_datamodule):
    dataset = dummy_datamodule.instantiate_dataset()
    assert isinstance(dataset, DummyDataset)


def test_setup():
    datamodule = DummyDatamodule()
    datamodule.setup('fit')

    assert isinstance(datamodule.train_dataset, Dataset)
    assert isinstance(datamodule.val_dataset, Dataset)

    datamodule.setup('test')

    assert isinstance(datamodule.train_dataset, Dataset)
    assert isinstance(datamodule.val_dataset, Dataset)
    assert isinstance(datamodule.test_dataset, Dataset)


def test_train_dataloader(dummy_datamodule):
    train_dataloader = dummy_datamodule.train_dataloader()
    assert isinstance(train_dataloader, DataLoader)

    inputs, targets = next(iter(train_dataloader))
    x, y = inputs['x'], targets['y']
    assert_close(x, torch.zeros_like(x))
    assert y == 1


def test_val_dataloader(dummy_datamodule):
    val_dataloader = dummy_datamodule.val_dataloader()
    assert isinstance(val_dataloader, DataLoader)

    inputs, targets = next(iter(val_dataloader))
    x, y = inputs['x'], targets['y']
    assert_close(x, torch.full_like(x, 0.4))
    assert y == 1


def test_test_dataloader(dummy_datamodule):
    test_dataloader = dummy_datamodule.test_dataloader()
    assert isinstance(test_dataloader, DataLoader)

    inputs, targets = next(iter(test_dataloader))
    x, y = inputs['x'], targets['y']
    assert_close(x, torch.full_like(x, 0.6))
    assert y == 1
