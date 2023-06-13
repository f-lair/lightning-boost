import models
import systems
from data import datamodules, datasets
from modules import loss, metrics

from lightning_boost.cli import LightningBoostCLI


def main():
    cli = LightningBoostCLI()


if __name__ == "__main__":
    main()
