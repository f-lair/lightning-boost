# Copyright 2023 Fabrice von der Lehr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
