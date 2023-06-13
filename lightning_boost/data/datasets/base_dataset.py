from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import inflection
from torch.utils.data import Dataset

from lightning_boost.modules.preprocessing import BaseTransform


class BaseDataset(Dataset):
    """Base class for dataset."""

    def __init__(
        self,
        root: str,
        download: bool = False,
        transform: Optional[BaseTransform] = None,
        **kwargs
    ) -> None:
        """
        Initiates base dataset.

        Args:
            root (str): Directory, where data is to be stored.
            download (bool, optional): Whether dataset should be downloaded. Defaults to False.
            transform (BaseTransform, optional): Data transform. Defaults to None.
        """

        self.path = Path.joinpath(Path(root), Path(inflection.underscore(type(self).__name__)))
        self.transform = transform
        if download:
            self._download()

    def get_item(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns data item at given index.

        Args:
            index (int): Index.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete dataset.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Inputs, targets.
        """

        raise NotImplementedError

    def __len__(self) -> int:
        """
        Returns size of dataset.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete dataset.

        Returns:
            int: Dataset size.
        """

        raise NotImplementedError

    def download(self) -> None:
        """
        Downloads dataset.
        """

        pass

    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns (transformed) data item at given index.

        Args:
            index (int): Index.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Inputs, targets.
        """

        inputs, targets = self.get_item(index)

        if self.transform is not None:
            inputs, targets = self.transform(inputs, targets)

        return inputs, targets

    def _download(self) -> None:
        """Downloads dataset (closure function)."""

        if Path.exists(self.path):
            return

        self.path.mkdir(parents=True, exist_ok=True)

        self.download()
