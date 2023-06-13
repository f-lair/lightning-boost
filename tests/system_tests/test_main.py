import subprocess
import sys

import pytest


def test_main():
    subprocess.run(
        [sys.executable, "./system_tests/main.py", "fit", "--config=./system_tests/config.yaml"]
    )
    subprocess.run(
        [
            sys.executable,
            "./system_tests/main.py",
            "validate",
            "--ckpt_path=./lightning_logs/dummy_test/checkpoints/last-epoch=1-step=8.ckpt",
            "--config=./system_tests/config.yaml",
        ]
    )
    subprocess.run(
        [
            sys.executable,
            "./system_tests/main.py",
            "test",
            "--ckpt_path=./lightning_logs/dummy_test/checkpoints/last-epoch=1-step=8.ckpt",
            "--config=./system_tests/config.yaml",
        ]
    )


if __name__ == "__main__":
    test_main()
