from typing import Any

from torchmetrics import Metric


class TaskMetric(Metric):
    """Wrapper-class for task-specific metrics."""

    def __init__(self, instance: Metric, task: str = 'base-task') -> None:
        """
        Initiates task-specific metric.

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
