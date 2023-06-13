from datetime import datetime

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from lightning_boost.data.datamodules import BaseDatamodule
from lightning_boost.modules.loss import TaskLoss
from lightning_boost.modules.metrics import TaskMetric
from lightning_boost.systems import BaseSystem


class LightningBoostCLI(LightningCLI):
    """Boosted command line interface."""

    def __init__(self, **kwargs) -> None:
        """Initiates command line interface."""

        super().__init__(
            model_class=BaseSystem,
            datamodule_class=BaseDatamodule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            auto_configure_optimizers=False,
            save_config_kwargs={"overwrite": True},
        )

    def add_core_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """
        Adds arguments from the core classes to the parser.

        Args:
            parser (LightningArgumentParser): Argument parser.
        """

        # adapted from pytorch-lightning implementation:
        # - replaced 'model' by 'system'
        # - removed if condition for datamodule
        parser.add_lightning_class_args(self.trainer_class, 'trainer')
        trainer_defaults = {
            'trainer.' + k: v for k, v in self.trainer_defaults.items() if k != 'callbacks'
        }
        parser.set_defaults(trainer_defaults)

        parser.add_lightning_class_args(
            self._model_class, 'system', subclass_mode=self.subclass_mode_model
        )
        parser.add_lightning_class_args(
            self._datamodule_class, 'data', subclass_mode=self.subclass_mode_data
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """
        Adds additional arguments to parser.
        Manages default arguments and argument linkings.

        Args:
            parser (LightningArgumentParser): Argument parser.
        """

        ### defaults ###
        # tensorboard logger
        parser.set_defaults(
            {
                'trainer.logger': {
                    'class_path': 'lightning.pytorch.loggers.TensorBoardLogger',
                    'init_args': {
                        'save_dir': 'lightning_logs',
                        'name': None,
                        'version': 'run__%d_%m_%Y__%H_%M_%S',
                    },
                }
            }
        )
        # last checkpoint
        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint_last')
        parser.set_defaults(
            {
                'model_checkpoint_last.monitor': None,
                'model_checkpoint_last.filename': 'last-{epoch}-{step}',
            }
        )
        # min loss checkpoint
        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint_min_loss')
        parser.set_defaults(
            {
                'model_checkpoint_min_loss.monitor': 'val_total-loss',
                'model_checkpoint_min_loss.filename': 'min-loss-{epoch}-{step}',
                'model_checkpoint_min_loss.mode': 'min',
            }
        )
        # system compilation
        parser.add_argument("--compile_system", default=False)

        ### linking ###
        # number of gpus
        parser.link_arguments(
            ('trainer.accelerator', 'trainer.devices'),
            'data.init_args.num_gpus',
            self.get_num_gpus,
        )

    def instantiate_classes(self) -> None:
        if str(self.subcommand) == 'predict':
            raise ValueError("Subcommand 'predict' is not supported yet!")

        config = self.config.get(str(self.subcommand), self.config)
        logger_version = config.get('trainer.logger.init_args.version', None)
        if logger_version is not None:
            config['trainer.logger.init_args.version'] = datetime.now().strftime(logger_version)

        # adapted from pytorch-lightning implementation, replaced 'model' by 'system'
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, 'data')
        self.model = self._get(self.config_init, 'system')
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()

    def fit(self, model: LightningModule, **kwargs) -> None:
        """
        Extends fit subcommand by optional system compilation.
        cf. https://github.com/Lightning-AI/lightning/issues/17283#issuecomment-1501890603

        Args:
            model (LightningModule): System to be trained (has to be named 'model').
        """

        system = model
        config = self.config.get(str(self.subcommand), self.config)

        if config["compile_system"]:
            compiled_system = torch.compile(system)
            self.trainer.fit(compiled_system, **kwargs)  # type: ignore
        else:
            self.trainer.fit(system, **kwargs)

    @staticmethod
    def get_num_gpus(accelerator: str, devices: int) -> int:
        """
        Returns number of GPUs.

        Args:
            accelerator (str): Accelerator (e.g., cpu/gpu).
            devices (int): Number of devices.

        Returns:
            int: Number of GPUs.
        """

        return devices if accelerator == "gpu" else 0
