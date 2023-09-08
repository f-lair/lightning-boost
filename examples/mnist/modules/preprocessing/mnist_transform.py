from typing import Dict, List, Tuple

import torch
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms.functional import normalize, to_tensor

from lightning_boost.modules.preprocessing import BaseTransform, CompositeTransform


# --8<-- [start:mnist_transform]
class MNISTTransform(BaseTransform):
    """Composite transform for MNIST classification task."""

    def __init__(self) -> None:
        """Initializes transform for MNIST dataset."""

        self.transform = CompositeTransform(
            [ToTensorTransform(), StandardizeTransform([0.1307], [0.3081])]
        )

    def __call__(
        self, inputs: Dict[str, Image], targets: Dict[str, int]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Performs transform for MNIST data.

        Args:
            inputs (Dict[str, Image]): Inputs.
            targets (Dict[str, int]): Targets.

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]: Inputs, targets (transformed).
        """

        return self.transform(inputs, targets)


# --8<-- [end:mnist_transform]


# --8<-- [start:to_tensor_transform]
class ToTensorTransform(BaseTransform):
    """Transform into tensors."""

    def __call__(
        self, inputs: Dict[str, Image], targets: Dict[str, int]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Transforms PIL image (input) and integer (target) into tensors.

        Args:
            inputs (Dict[str, Image]): Inputs.
            targets (Dict[str, int]): Targets.

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]: Inputs, targets (transformed).
        """

        inputs["x"] = to_tensor(inputs["x"])
        targets["y"] = torch.tensor(targets["y"], dtype=torch.long)

        return inputs, targets


# --8<-- [end:to_tensor_transform]


# --8<-- [start:standardize_transform]
class StandardizeTransform(BaseTransform):
    """Standardization transform."""

    def __init__(self, mean: List[float], std: List[float]) -> None:
        """
        Initializes transform.

        Args:
            mean (List[float]): Channel-wise means of input data.
            std (List[float]): Channel-wise standard deviations of input data.
        """

        self.mean = mean
        self.std = std

    def __call__(
        self, inputs: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Standardizes input tensor.

        Args:
            inputs (Dict[str, Any]): Inputs.
            targets (Dict[str, Any]): Targets.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Inputs, targets (transformed).
        """

        normalize(inputs["x"], mean=self.mean, std=self.std, inplace=True)  # (1)

        return inputs, targets


# --8<-- [end:standardize_transform]
