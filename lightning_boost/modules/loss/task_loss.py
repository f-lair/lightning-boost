import torch
from torch.nn import Module


class TaskLoss(Module):
    """Wrapper class for task-specific loss functions."""

    def __init__(self, instance: Module, task: str = 'base-task', weight: float = 1.0) -> None:
        """
        Initiates task-specific loss function.

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
