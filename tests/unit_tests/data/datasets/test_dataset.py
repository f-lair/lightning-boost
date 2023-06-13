from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from lightning_boost.data.datasets import BaseDataset
from tests.dummy.data.datasets import DummyDataset
from tests.dummy.modules.preprocessing import DummyTransform


def test_init():
    dataset = BaseDataset('./data/download')
    with pytest.raises(NotImplementedError) as exc_info:
        _ = len(dataset)
    with pytest.raises(NotImplementedError) as exc_info:
        _ = dataset[0]

    dataset = DummyDataset('./data/download')
    assert str(dataset.path) == str(Path('data/download/dummy_dataset'))
    assert dataset.transform is None
    assert dataset.downloaded == False

    transform = DummyTransform()
    dataset = DummyDataset('./data/download', transform=transform, download=True)
    assert str(dataset.path) == str(Path('data/download/dummy_dataset'))
    assert dataset.transform == transform
    assert dataset.downloaded == True


def test_get_item(dummy_datamodule):
    size = (2, 42)

    # without DummyTransform
    dataset = DummyDataset('./data/download', size=size)
    inputs, targets = dataset.get_item(1)
    x, y = inputs['x'], targets['y']
    assert_close(x, torch.ones_like(x))
    assert y == 0

    # with DummyTransform
    dataset = dummy_datamodule.instantiate_dataset(size=size)
    inputs, targets = dataset.get_item(1)
    x, y = inputs['x'], targets['y']
    assert_close(x, torch.ones_like(x))
    assert y == 0


def test_len(dummy_datamodule):
    N = 1
    dataset = dummy_datamodule.instantiate_dataset(size=(N, 42))
    assert len(dataset) == N

    N = 2
    dataset = dummy_datamodule.instantiate_dataset(size=(N, 42))
    assert len(dataset) == N


def test_download():
    dataset = BaseDataset('./data/download')
    dataset.download()

    dataset = DummyDataset('./data/download', download=False)
    assert dataset.downloaded == False

    dataset = DummyDataset('./data/download', download=True)
    assert dataset.downloaded == True

    dataset = DummyDataset('./data/download', download=True, remove_root_dir=False)
    assert dataset.downloaded == False


def test_get_item_2(dummy_datamodule):
    size = (2, 42)

    # without DummyTransform
    dataset = DummyDataset('./data/download', size=size)
    inputs, targets = dataset[1]
    x, y = inputs['x'], targets['y']
    assert_close(x, torch.ones_like(x))
    assert y == 0

    # with DummyTransform
    dataset = dummy_datamodule.instantiate_dataset(size=size)
    inputs, targets = dataset[1]
    x, y = inputs['x'], targets['y']
    assert_close(x, torch.full_like(x, 0.1))
    assert y == 1
