import pytest

from lightning_boost.cli import LightningBoostCLI


def test_init():
    with pytest.raises(SystemExit) as exc_info:
        cli = LightningBoostCLI()
    assert exc_info.type == SystemExit
    assert exc_info.value.code == 2


def test_get_num_gpus():
    assert LightningBoostCLI.get_num_gpus(accelerator='cpu', devices=0) == 0
    assert LightningBoostCLI.get_num_gpus(accelerator='cpu', devices=42) == 0
    assert LightningBoostCLI.get_num_gpus(accelerator='gpu', devices=0) == 0
    assert LightningBoostCLI.get_num_gpus(accelerator='gpu', devices=42) == 42
