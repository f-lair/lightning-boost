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
