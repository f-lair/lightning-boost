import torch
from torch import Tensor
from torch.nn import Linear, Module, Sigmoid

from lightning_boost.models import BaseModel


class DummyModel(BaseModel):
    def __init__(self, size_in: int = 42, size_out: int = 1) -> None:
        super().__init__()
        self.lin = Linear(size_in, size_out)
        self.sigmoid = Sigmoid()

        self.lin.apply(DummyModel.init_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(self.lin(x))[:, 0]

    @torch.no_grad()
    @staticmethod
    def init_weights(m: Module) -> None:
        # deterministic weights
        if isinstance(m, Linear):
            m.weight[:] = 0.0
            m.bias[:] = 0.0
