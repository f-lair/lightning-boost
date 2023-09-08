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


from typing import Any, Dict, Tuple


class BaseTransform:
    """Base class for transform."""

    def __call__(
        self, inputs: Dict[str, Any], targets: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Performs transform.

        Args:
            inputs (Dict[str, Any]): Inputs.
            targets (Dict[str, Any]): Targets.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Inputs, targets (transformed).
        """

        raise NotImplementedError
