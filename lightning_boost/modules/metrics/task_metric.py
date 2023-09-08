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


from typing import Any

from torchmetrics import Metric


class TaskMetric(Metric):
    """Wrapper-class for task-specific metrics."""

    def __init__(self, instance: Metric, task: str = 'base-task') -> None:
        """
        Initializes task-specific metric.

        Args:
            instance (Metric): Metric.
            task (str, optional): Task. Defaults to 'base-task'.
        """

        super().__init__()
        self.instance = instance
        self.task = task

    def update(self, *args, **kwargs) -> None:
        """
        Performs metric update.
        """

        self.instance.update(*args, **kwargs)

    def compute(self) -> Any:
        """
        Evaluates metric.

        Returns:
            Any: Metric value.
        """

        return self.instance.compute()
