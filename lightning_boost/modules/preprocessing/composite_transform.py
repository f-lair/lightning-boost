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
