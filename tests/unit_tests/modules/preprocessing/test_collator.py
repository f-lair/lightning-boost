import pytest
import torch
from torch.testing import assert_close
from torch.utils.data import DataLoader
from utils import assertDictEqual

from lightning_boost.modules.preprocessing import BaseCollator
from tests.dummy.data.datamodules import DummyDatamodule
from tests.dummy.modules.preprocessing import DummyCollator


def test_init():
    with pytest.raises(NotImplementedError) as exc_info:
        collator = BaseCollator()

    collator = DummyCollator()
    assert collator.pad_val == 0
    assert collator.pad_shape == []
    assert collator.pad_dims == []

    pad_val = 42
    pad_shape = [2, 4, 8]
    pad_dims = [1, 2, 3]
    collator = DummyCollator(pad_val=pad_val, pad_shape=pad_shape, pad_dims=pad_dims)
    assert collator.pad_val == pad_val
    assert collator.pad_shape == pad_shape
    assert collator.pad_dims == pad_dims

    pad_shape = [2, 4, 8]
    pad_dims = [1, 2]
    with pytest.raises(AssertionError) as exc_info:
        collator = DummyCollator(pad_val=pad_val, pad_shape=pad_shape, pad_dims=pad_dims)


def test_call():
    collator = DummyCollator()
    N, D = 4, 42
    batch = [({'x': torch.full((D,), val)}, {'y': torch.ones(())}) for val in range(N)]
    collated_batch = collator(batch)
    collated_batch_expected = (
        {'x': torch.arange(0, N, 1).unsqueeze(1).expand((N, D))},
        {'y': torch.ones((N,))},
    )

    assertDictEqual(collated_batch[0], collated_batch_expected[0])
    assertDictEqual(collated_batch[1], collated_batch_expected[1])


def test_pad_collate_nd():
    collator = DummyCollator()

    # Test 2d
    shapes = [(4, 3), (3, 4)]
    batch = []
    for idx in range(len(shapes)):
        batch.append(torch.ones((shapes[idx])))
    out = collator.pad_collate_nd(batch)
    out_ref = torch.ones((len(shapes), 4, 4))
    out_ref[0, :, -1] = 0.0
    out_ref[1, -1, :] = 0.0
    assert_close(out, out_ref)

    # Test 3d
    shapes = [(6, 4, 2), (1, 3, 5)]
    batch = []
    for idx in range(len(shapes)):
        batch.append(torch.ones((shapes[idx])))
    out = collator.pad_collate_nd(batch)
    out_ref = torch.ones((len(shapes), 6, 4, 5))
    out_ref[0, :, :, -3:] = 0.0
    out_ref[1, -5:, :, :] = 0.0
    out_ref[1, :, -1:, :] = 0.0
    assert_close(out, out_ref)

    # Test 3d + custom pad dims
    collator = DummyCollator(pad_shape=[42], pad_dims=[1])
    shapes = [(6, 4, 2), (1, 3, 5)]
    batch = []
    for idx in range(len(shapes)):
        batch.append(torch.ones((shapes[idx])))
    out = collator.pad_collate_nd(batch)
    out_ref = torch.ones((len(shapes), 6, 42, 5))
    out_ref[0, :, 4:, :] = 0.0
    out_ref[0, :, :, -3:] = 0.0
    out_ref[1, -5:, :, :] = 0.0
    out_ref[1, :, 3:, :] = 0.0
    assert_close(out, out_ref)

    # Test dedicated worker process
    dummy_datamodule = DummyDatamodule(batch_size=2, num_workers=1)
    dummy_datamodule.setup('fit')
    dummy_datamodule.setup('test')
    test_dataloader = dummy_datamodule.test_dataloader()
    inputs, _ = next(iter(test_dataloader))
    x = inputs['x']
    assert x.shape == (2, 42)


def test_flatten_collate(dummy_datamodule):
    collator = DummyCollator()

    N = 10
    batch = [torch.full((), val) for val in range(N)]
    out = collator.flatten_collate(batch)
    out_ref = torch.arange(0, N, 1)
    assert_close(out, out_ref)

    # Test dedicated worker process
    dummy_datamodule = DummyDatamodule(batch_size=2, num_workers=1)
    dummy_datamodule.setup('fit')
    dummy_datamodule.setup('test')
    test_dataloader = dummy_datamodule.test_dataloader()
    _, targets = next(iter(test_dataloader))
    y = targets['y']
    assert y.shape == (2,)
