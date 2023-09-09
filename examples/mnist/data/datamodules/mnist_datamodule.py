from typing import Tuple, Type

from data.datasets import MNISTDataset
from modules.preprocessing import MNISTCollator, MNISTTransform
from torch.utils.data import Dataset, random_split

from lightning_boost.data.datamodules import BaseDatamodule


# --8<-- [start:mnist_datamodule]
class MNISTDatamodule(BaseDatamodule):
    """Datamodule for MNIST classification task."""

    def get_collator(self, **kwargs) -> MNISTCollator:
        """
        Returns collator, which transforms a list of batch items to a tensor with additional
        batch dimension.

        Returns:
            MNISTCollator: Collator.
        """

        return MNISTCollator()

    def get_dataset_type(self) -> Type[MNISTDataset]:
        """
        Returns type of used dataset.

        Returns:
            Type[MNISTDataset]: Dataset type.
        """

        return MNISTDataset

    def get_transform(self, **kwargs) -> MNISTTransform:
        """
        Returns transform, which preprocesses data.

        Returns:
            MNISTTransform: Transform.
        """

        return MNISTTransform()

    def get_train_test_split(self) -> Tuple[Dataset, Dataset]:
        """
        Returns train (incl. validation data) and test datasets (only for cross-validation).

        Returns:
            Tuple[Dataset, Dataset]: Train, test dataset.
        """

        train_dataset = self.instantiate_dataset(train=True)
        test_dataset = self.instantiate_dataset(train=False)

        return train_dataset, test_dataset


# --8<-- [end:mnist_datamodule]
