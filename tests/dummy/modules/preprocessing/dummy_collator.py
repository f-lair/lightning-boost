from typing import Callable, Dict, List

from torch import Tensor

from lightning_boost.modules.preprocessing import BaseCollator


class DummyCollator(BaseCollator):
    def get_collate_fn(self) -> Dict[str, Callable[[List[Tensor]], Tensor]]:
        return {
            "x": self.pad_collate_nd,
            "y": self.flatten_collate,
        }
