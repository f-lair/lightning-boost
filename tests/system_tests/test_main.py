import subprocess
import sys

import pytest


def test_main():
    completed_process = subprocess.run(
        [
            sys.executable,
            "./tests/system_tests/main.py",
            "fit",
            "--config=./tests/system_tests/config.yaml",
        ]
    )
    assert completed_process.returncode == 0

    completed_process = subprocess.run(
        [
            sys.executable,
            "./tests/system_tests/main.py",
            "validate",
            "--ckpt_path=./tests/lightning_logs/dummy_test/checkpoints/last-epoch=1-step=8.ckpt",
            "--config=./tests/system_tests/config.yaml",
        ]
    )
    assert completed_process.returncode == 0

    completed_process = subprocess.run(
        [
            sys.executable,
            "./tests/system_tests/main.py",
            "test",
            "--ckpt_path=./tests/lightning_logs/dummy_test/checkpoints/last-epoch=1-step=8.ckpt",
            "--config=./tests/system_tests/config.yaml",
        ]
    )
    assert completed_process.returncode == 0


if __name__ == "__main__":
    test_main()
