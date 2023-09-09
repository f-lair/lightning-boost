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


from typing import Any, List

import inflection
from torch.nn import ModuleDict

from lightning_boost.models import BaseModel
from lightning_boost.modules.loss import TaskLoss
from lightning_boost.modules.metrics import TaskMetric


def dasherize_class_name(instance: Any) -> str:
    """
    Dasherizes class name of passed instance.
    E.g., 'LightningBoost' -> 'lightning-boost'

    Args:
        instance (Any): Instance, whose class name is to be dasherized.

    Returns:
        str: Dasherized class name.
    """

    return inflection.dasherize(inflection.underscore(type(instance).__name__))


def get_loss_dict(loss: List[TaskLoss] | TaskLoss) -> ModuleDict:
    """
    Returns dictionary of (task, (name, loss)) pairs for a potential list of loss functions.

    Args:
        loss (List[TaskLoss] | TaskLoss): Loss function(s).

    Returns:
        ModuleDict[str, ModuleDict[str, Module]]: Task, name, loss function.
    """

    if isinstance(loss, TaskLoss):
        loss = [loss]

    loss_dict = ModuleDict()
    for task_loss in loss:
        if task_loss.task not in loss_dict:
            loss_dict[task_loss.task] = ModuleDict()
        loss_dict[task_loss.task][dasherize_class_name(task_loss.instance)] = task_loss  # type: ignore

    return loss_dict


def get_metrics_dict(
    metrics: List[TaskMetric] | TaskMetric | None,
) -> ModuleDict:
    """
    Returns dictionary of (task, (name, metric)) pairs for a list of metrics.

    Args:
        metrics (List[TaskMetric] | TaskMetric | None): Metric(s).

    Returns:
        ModuleDict[str, ModuleDict[str, Metric]]: Task, name, metric.
    """

    if metrics is None:
        return ModuleDict({'base-task': ModuleDict()})
    else:
        if isinstance(metrics, TaskMetric):
            metrics = [metrics]
        metrics_dict = ModuleDict()
        for task_metric in metrics:
            if task_metric.task not in metrics_dict:
                metrics_dict[task_metric.task] = ModuleDict()
            metrics_dict[task_metric.task][dasherize_class_name(task_metric.instance)] = task_metric  # type: ignore

        return metrics_dict


def get_models_dict(
    models: List[BaseModel] | BaseModel,
) -> ModuleDict:
    """
    Returns dictionary of (name, model) pairs for a list of DL models.

    Args:
        models (List[BaseModel] | BaseModel): DL model(s).

    Returns:
        ModuleDict[str, Metric]: Task, DL model.
    """

    if isinstance(models, BaseModel):
        models = [models]

    return ModuleDict(
        {
            dasherize_class_name(model) if model.name is None else model.name: model
            for model in models
        }
    )
