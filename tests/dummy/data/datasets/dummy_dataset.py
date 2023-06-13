import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

from lightning_boost.data.datasets import BaseDataset
from lightning_boost.modules.preprocessing import BaseTransform


class DummyDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        download: bool = False,
        transform: Optional[BaseTransform] = None,
        size: Sequence[int] = (8, 42),
        remove_root_dir=True,
        **kwargs,
    ) -> None:
        self.downloaded = False

        # Remove root such that tests can succeed
        root_path = Path(root)
        if remove_root_dir and root_path.exists() and root_path.is_dir():
            shutil.rmtree(root_path)

        super().__init__(root, download, transform)

        self.size = tuple(size)
        self.dataset_x = (
            torch.arange(0, self.size[0], 1, dtype=torch.float32)
            .view(self.size[0:1] + (1,) * (len(self.size) - 1))
            .expand(self.size)
        )
        self.dataset_y = torch.zeros(self.size[0:1])

    def download(self) -> None:
        self.downloaded = True

    def get_item(self, index: int) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        x, y = self.dataset_x[index], self.dataset_y[index]

        return {"x": x}, {"y": y}

    def __len__(self) -> int:
        return self.size[0]
