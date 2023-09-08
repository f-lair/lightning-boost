from typing import Dict, Optional, Tuple

from PIL.Image import Image
from torchvision.datasets import MNIST

from lightning_boost.data.datasets import BaseDataset
from lightning_boost.modules.preprocessing import BaseTransform


class MNISTDataset(BaseDataset):
    """MNIST dataset."""

    # --8<-- [start:init]
    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        transform: Optional[BaseTransform] = None,
        **kwargs
    ) -> None:
        """
        Initializes MNIST dataset.

        Args:
            root (str): Directory, where data is to be stored.
            train (bool, optional): Whether training data is used. Defaults to True.
            download (bool, optional): Whether dataset should be downloaded. Defaults to False.
            transform (BaseTransform, optional): Data transform. Defaults to None.
        """

        super().__init__(root, download, transform)  # (1)

        self.train = train
        self.dataset = MNIST(str(self.path), train=self.train, download=False)  # (2)

    # --8<-- [end:init]

    # --8<-- [start:download]
    def download(self) -> None:
        """
        Downloads dataset.
        """

        MNIST(str(self.path), train=True, download=True)
        MNIST(str(self.path), train=False, download=True)

    # --8<-- [end:download]

    # --8<-- [start:get_item]
    def get_item(self, index: int) -> Tuple[Dict[str, Image], Dict[str, int]]:
        """
        Returns data item at given index.

        Args:
            index (int): Index.

        Returns:
            Tuple[Dict[str, Image], Dict[str, int]]: Inputs, targets.
        """

        img, target = self.dataset[index]

        return {"x": img}, {"y": target}

    # --8<-- [end:get_item]

    # --8<-- [start:len]
    def __len__(self) -> int:
        """
        Returns size of dataset.

        Returns:
            int: Dataset size.
        """

        return len(self.dataset)

    # --8<-- [end:len]
