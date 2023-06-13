from typing import Dict

from torch import Tensor
from torch.nn import ModuleDict
from torch.testing import assert_close


def assertDictEqual(actual: Dict | ModuleDict, expected: Dict | ModuleDict) -> None:
    """
    Checks whether two dictionaries are equal in terms of their (key, value) pairs.
    Takes torch.Tensor closeness and nested dictionaries into account.

    Args:
        actual (Dict | ModuleDict): First dictionary.
        expected (Dict | ModuleDict): Second dictionary.
    """

    # check for equality of key sets
    assert len(set(actual.keys()) - set(expected.keys())) == 0
    assert len(set(expected.keys()) - set(actual.keys())) == 0

    # check for equality/closeness of value elements
    for key in actual.keys():
        if (isinstance(actual[key], dict) or isinstance(actual[key], ModuleDict)) and (
            isinstance(expected[key], dict) or isinstance(expected[key], ModuleDict)
        ):
            assertDictEqual(actual[key], expected[key])  # type: ignore
        elif isinstance(actual[key], Tensor) and isinstance(expected[key], Tensor):
            assert_close(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]
