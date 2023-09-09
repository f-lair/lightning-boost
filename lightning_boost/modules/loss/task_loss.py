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


import torch
from torch.nn import Module


class TaskLoss(Module):
    """Wrapper class for task-specific loss functions."""

    def __init__(self, instance: Module, task: str = 'base-task', weight: float = 1.0) -> None:
        """
        Initializes task-specific loss function.

        Args:
            instance (Module): Loss function.
            task (str, optional): Task. Defaults to 'base-task'.
            weight (float, optional): Weight in sum of all loss functions. Defaults to 1..
        """

        super().__init__()
        self.instance = instance
        self.task = task
        self.weight = weight

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluates loss function.

        Args:
            y_hat (torch.Tensor): Prediction.
            y (torch.Tensor): Target.

        Returns:
            torch.Tensor: Loss.
        """

        return self.instance(y_hat, y)
