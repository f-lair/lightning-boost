from torch import Tensor
from torch.nn import Flatten, Linear, ReLU

from lightning_boost.models import BaseModel


# --8<-- [start:mnist_model]
class MNISTModel(BaseModel):
    """MLP model for MNIST classification task."""

    def __init__(self) -> None:
        """
        Initiates FC-784-256-256-10 model.
        """

        super().__init__()
        self.flatten = Flatten()
        self.lin1 = Linear(784, 256)
        self.lin2 = Linear(256, 256)
        self.lin3 = Linear(256, 10)
        self.act = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass.
        B: Batch dim.
        P: Pixels per dim (28).
        C: Class dim (10).

        Args:
            x (Tensor): Input [B, 1, P, P].

        Returns:
            Tensor: Prediction [B, C].
        """

        out = self.flatten(x)
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        out = self.act(out)
        out = self.lin3(out)

        return out


# --8<-- [end:mnist_model]
