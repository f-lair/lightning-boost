# Copyright 2023 Fabrice von der Lehr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, List, Tuple

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from lightning_boost.models import BaseModel
from lightning_boost.modules.loss import TaskLoss
from lightning_boost.modules.metrics import TaskMetric
from lightning_boost.modules.optim import LRSchedulingPolicy
from lightning_boost.systems.utils import (
    get_loss_dict,
    get_metrics_dict,
    get_models_dict,
)


class BaseSystem(LightningModule):
    """Base class for DL system."""

    def __init__(
        self,
        models: List[BaseModel] | BaseModel,
        loss: List[TaskLoss] | TaskLoss,
        optimizer: OptimizerCallable,
        lr_scheduler: LRSchedulerCallable | None = None,
        lr_scheduling_policy: LRSchedulingPolicy = LRSchedulingPolicy(),
        train_metrics: List[TaskMetric] | TaskMetric | None = None,
        val_metrics: List[TaskMetric] | TaskMetric | None = None,
        test_metrics: List[TaskMetric] | TaskMetric | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes DL system.

        Args:
            models (List[BaseModel]): DL model(s).
            loss (List[TaskLoss] | TaskLoss): Loss function(s).
            optimizer (OptimizerCallable): Optimizer.
            lr_scheduler (LRSchedulerCallable | None, optional): Learning rate scheduler. Defaults to None.
            lr_scheduling_policy (LRSchedulingPolicy | None, optional): Learning rate scheduling policy. Defaults to None.
            train_metrics (List[TaskMetric] | TaskMetric | None, optional): Metric(s) used for training. Defaults to None.
            val_metrics (List[TaskMetric] | TaskMetric | None, optional): Metric(s) used for validation. Defaults to None.
            test_metrics (List[TaskMetric] | TaskMetric | None, optional): Metric(s) used for testing. Defaults to None.
        """

        super().__init__()

        self.models = get_models_dict(models)
        self.loss_functions = get_loss_dict(loss)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduling_policy = lr_scheduling_policy
        self.train_metrics = get_metrics_dict(train_metrics)
        self.val_metrics = get_metrics_dict(val_metrics)
        self.test_metrics = get_metrics_dict(test_metrics)

    def step(self, inputs: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Performs a single step in training/validation/testing.

        Args:
            inputs (Dict[str, Tensor]): Inputs.
            targets (Dict[str, Tensor]): Targets.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete DL system.

        Returns:
            Dict[str, Tensor]: Predictions.
        """

        raise NotImplementedError

    def _base_step(
        self,
        batch_data: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int,
        test: bool = False,
    ) -> Tensor:
        """
        Closure function for steps in all modes.

        Args:
            batch_data (Tuple[Dict[str, Tensor], Dict[str, Tensor]]): Inputs, targets.
            batch_idx (int): Batch index in epoch.
            test (bool, optional): Whether testing mode is active. Defaults to False.

        Returns:
            Tensor: Loss.
        """

        inputs, targets = batch_data
        assert not (self.training and test)

        if self.training:
            mode = 'train'
            metrics = self.train_metrics
        else:
            mode = 'val'
            metrics = self.val_metrics
        if test:
            mode = 'test'
            metrics = self.test_metrics

        predictions = self.step(inputs, targets)
        losses = {}

        # all keys in predictions have to be existent in targets
        assert (
            len(set(predictions.keys()) - set(targets.keys())) == 0
        ), "Keys of predictions and targets muss match!"
        # with multiple predictions, this also holds for loss_functions and metrics
        if len(predictions) > 1:
            assert (
                len(set(predictions.keys()) - set(self.loss_functions.keys())) == 0
            ), "Keys of predictions and loss functions must match!"
            assert (
                len(set(predictions.keys()) - set(metrics.keys())) == 0
            ), "Keys of predictions and metrics must match!"

            for task in predictions:
                for metric_name, metric in metrics[task].items():  # type: ignore
                    label = f"{task}_{mode}_{metric_name}"
                    metric(predictions[task], targets[task])
                    self.log(
                        label,
                        metric,  # type: ignore
                        on_step=self.training,
                        on_epoch=True,
                        sync_dist=True,
                    )
                for loss_name, loss_function in self.loss_functions[task].items():  # type: ignore
                    label = f"{task}_{mode}_{loss_name}"
                    losses[label] = (
                        loss_function(predictions[task], targets[task]),
                        loss_function.weight,
                    )
                    self.log(
                        label,
                        losses[label][0],
                        on_step=self.training,
                        on_epoch=True,
                        sync_dist=True,
                    )
            # compute weighted loss over losses
            loss = torch.sum(
                torch.stack(
                    [weight * task_loss for weight, task_loss in losses.values()],
                    dim=0,
                ),
                dim=0,
            )

            self.log(
                f"{mode}_total-loss",
                loss,
                on_step=self.training,
                on_epoch=True,
                sync_dist=True,
            )
        else:
            loss_functions = next(iter(self.loss_functions.values()))
            metrics = next(iter(metrics.values()))
            y_hat, y = next(iter(predictions.values())), next(iter(targets.values()))
            # there must be at least one loss function
            assert len(loss_functions) > 0, "No loss function specified!"  # type: ignore

            for metric_name, metric in metrics.items():  # type: ignore
                metric(y_hat, y)
                self.log(
                    f"{mode}_{metric_name}",
                    metric,  # type: ignore
                    on_step=self.training,
                    on_epoch=True,
                    sync_dist=True,
                )
            for loss_name, loss_function in loss_functions.items():  # type: ignore
                label = f"{mode}_{loss_name}"
                losses[label] = (
                    loss_function(y_hat, y),
                    loss_function.weight,
                )
                self.log(
                    label,
                    losses[label][0],
                    on_step=self.training,
                    on_epoch=True,
                    sync_dist=True,
                )
            # compute weighted loss over losses
            loss = torch.sum(
                torch.stack(
                    [weight * task_loss for weight, task_loss in losses.values()],
                    dim=0,
                ),
                dim=0,
            )

            self.log(
                f"{mode}_total-loss",
                loss,
                on_step=self.training,
                on_epoch=True,
                sync_dist=True,
            )

        return loss

    def training_step(
        self,
        batch_data: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int,
    ) -> Tensor:
        """
        Performs step in training mode.

        Args:
            batch_data (Tuple[Dict[str, Tensor], Dict[str, Tensor]]): Inputs, targets.
            batch_idx (int): Batch index in epoch.

        Returns:
            Tensor: Loss.
        """

        return self._base_step(batch_data, batch_idx)

    def validation_step(
        self,
        batch_data: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int,
    ) -> Tensor:
        """
        Performs step in validation mode.

        Args:
            batch_data (Tuple[Dict[str, Tensor], Dict[str, Tensor]]): Inputs, targets.
            batch_idx (int): Batch index in epoch.

        Returns:
            Tensor: Loss.
        """

        return self._base_step(batch_data, batch_idx)

    def test_step(
        self,
        batch_data: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int,
    ) -> Tensor:
        """
        Performs step in testing mode.

        Args:
            batch_data (Tuple[Dict[str, Tensor], Dict[str, Tensor]]): Inputs, targets.
            batch_idx (int): Batch index in epoch.

        Returns:
            Tensor: Loss.
        """

        return self._base_step(batch_data, batch_idx, test=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures optimizer and learning rate scheduling.

        Returns:
            Dict[str, Any]: Optimizer, LR scheduling policy.
        """

        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return {'optimizer': optimizer}

        lr_scheduler = self.lr_scheduler(optimizer)
        lr_scheduling_policy = self.lr_scheduling_policy.bind_lr_scheduler(lr_scheduler)  # type: ignore

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduling_policy}
