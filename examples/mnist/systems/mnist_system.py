from typing import Dict

from torch import Tensor

from lightning_boost.systems import BaseSystem


# --8<-- [start:mnist_system]
class MNISTSystem(BaseSystem):
    """DL system for MNIST classification task."""

    def step(self, inputs: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Performs a single step in training/validation/testing.

        Args:
            inputs (Dict[str, Tensor]): Inputs.
            targets (Dict[str, Tensor]): Targets.


        Returns:
            Dict[str, Tensor]: Predictions.
        """

        x = inputs['x']
        y_hat = self.models['mnist-model'](x)

        return {'y': y_hat}


# --8<-- [end:mnist_system]
