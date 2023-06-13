from typing import Sequence

from lightning.pytorch import LightningModule
from torch import Tensor


class BaseModel(LightningModule):
    """Base class for DL model."""

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self.name = name

    def forward(self, *args: Tensor) -> Tensor | Sequence[Tensor]:
        """
        Performs forward pass.

        Args:
            Tensor: Inputs.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete DL model.

        Returns:
            Tensor | Sequence[Tensor]: Predictions.
        """

        raise NotImplementedError
