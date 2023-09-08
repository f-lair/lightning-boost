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


from typing import Any, Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LRSchedulingPolicy:
    """Defines a learning rate scheduling policy."""

    def __init__(
        self, interval: str = 'epoch', frequency: int = 1, monitor: str = 'val_total-loss'
    ) -> None:
        """
        Initializes LR scheduling policy.

        Args:
            interval (str, optional): Interval in which LR scheduling is performed ('step' or 'epoch'). Defaults to 'epoch'.
            frequency (int, optional): Frequency in which LR scheduling is performed. Defaults to 1.
            monitor (str, optional): Monitor metric. Defaults to 'val_total-loss'.
        """

        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

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
            'monitor': self.monitor,
        }
