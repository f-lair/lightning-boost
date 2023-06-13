from typing import Dict

from torch import Tensor

from lightning_boost.systems import BaseSystem


class DummySystem(BaseSystem):
    def step(self, inputs: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = inputs['x']
        y_hat = self.models['dummy-model'](x)

        return {key: y_hat for key in targets.keys()}
