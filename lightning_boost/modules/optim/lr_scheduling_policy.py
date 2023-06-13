from typing import Any, Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LRSchedulingPolicy:
    """Defines a learning rate scheduling policy."""

    def __init__(self, interval: str = 'epoch', frequency: int = 1) -> None:
        """
        Initiates LR scheduling policy.

        Args:
            interval (str, optional): Interval in which LR scheduling is performed ('step' or 'epoch'). Defaults to 'epoch'.
            frequency (int, optional): Frequency in which LR scheduling is performed. Defaults to 1.
        """

        self.interval = interval
        self.frequency = frequency

    def bind_lr_scheduler(self, lr_scheduler: LRScheduler) -> Dict[str, Any]:
        """
        Binds learning rate scheduler and returns scheduling policy.

        Args:
            lr_scheduler (LRScheduler): Learning rate scheduler.

        Returns:
            Dict[str, Any]: LR scheduling policy.
        """

        return {
            'scheduler': lr_scheduler,
            'interval': self.interval,
            'frequency': self.frequency,
        }
