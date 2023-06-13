from typing import Callable, Dict, List

from torch import Tensor

from lightning_boost.modules.preprocessing import BaseCollator


# --8<-- [start:mnist_collator]
class MNISTCollator(BaseCollator):
    """Collator for MNIST classification task."""

    def get_collate_fn(self) -> Dict[str, Callable[[List[Tensor]], Tensor]]:
        """
        Returns collator functions for each data type in inputs and targets.

        Returns:
            Dict[str, Callable[[List[Tensor]], Tensor]]: Collator functions that take a list of
            tensors and return a single tensor.
        """

        return {
            "x": self.pad_collate_nd,
            "y": self.flatten_collate,
        }


# --8<-- [start:mnist_collator]
