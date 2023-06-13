from lightning_boost.cli import LightningBoostCLI
from tests.dummy import models, systems
from tests.dummy.data import datamodules, datasets
from tests.dummy.modules import loss, metrics


def main():
    cli = LightningBoostCLI()


if __name__ == "__main__":
    main()
