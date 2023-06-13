from typing import Dict, Tuple

from torch import Tensor

from lightning_boost.modules.preprocessing import BaseTransform


class DummyTransform(BaseTransform):
    def __call__(
        self, inputs: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        return {'x': 0.1 * inputs['x']}, {'y': 1 + targets['y']}
