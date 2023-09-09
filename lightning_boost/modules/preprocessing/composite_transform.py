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


from typing import Any, Dict, List, Tuple

from lightning_boost.modules.preprocessing import BaseTransform


class CompositeTransform(BaseTransform):
    def __init__(self, transforms: List[BaseTransform]) -> None:
        self.transforms = transforms

    def __call__(
        self, inputs: Dict[str, Any], targets: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Performs the given transforms in the specified order for inputs and targets.

        Args:
            inputs (Dict[str, Any]): Inputs.
            targets (Dict[str, Any]): Targets.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Inputs, targets (transformed).
        """

        for transform in self.transforms:
            inputs, targets = transform(inputs, targets)

        return inputs, targets
