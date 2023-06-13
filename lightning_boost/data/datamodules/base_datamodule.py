from typing import Tuple, Type

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from lightning_boost.data.datasets import BaseDataset
from lightning_boost.modules.preprocessing import BaseCollator, BaseTransform


class BaseDatamodule(LightningDataModule):
    """Base class for datamodule."""

    def __init__(
        self,
        data_dir: str = './data/download/',
        batch_size: int = 32,
        num_workers: int = 0,
        num_gpus: int = 0,
        shuffle: bool = True,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        **kwargs
    ):
        """
        Initiates datamodule.

        Args:
            data_dir (str, optional): Directory, where dataset is stored. Defaults to './data/download/'.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of additional workers used for data loading. Defaults to 0.
            num_gpus (int, optional): Number of used GPUs. Defaults to 0.
            shuffle (bool, optional): Whether training dataset is shuffled. Defaults to True.
            val_ratio (float, optional): Ratio of validation split. Defaults to 0.1.
            test_ratio (float, optional): Ratio of validation split. Defaults to 0.1.
        """

        super().__init__()

        assert batch_size > 0, "Batchsize must be positive!"
        assert num_workers >= 0, "Number of workers must be non-negative!"
        assert num_gpus >= 0, "Number of GPUs must be non-negative!"
        assert 0.0 <= val_ratio <= 1.0, "Validation ratio must be in [0, 1]!"
        assert 0.0 <= test_ratio <= 1.0, "Test ratio must be in [0, 1]!"
        assert (
            0.0 <= val_ratio + test_ratio <= 1.0
        ), "Sum of validation and test ratio must be in [0, 1]!"

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.shuffle = shuffle
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.collator = self.get_collator(**kwargs)
        self.dataset_type = self.get_dataset_type(**kwargs)
        self.transform = self.get_transform(**kwargs)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_collator(self, **kwargs) -> BaseCollator:
        """
        Returns collator, which transforms a list of batch items to a tensor with additional
        batch dimension.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete datamodule.

        Returns:
            BaseCollator: Collator.
        """

        raise NotImplementedError

    def get_dataset_type(self, **kwargs) -> Type[BaseDataset]:
        """
        Returns type of used dataset.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete datamodule.

        Returns:
            Type[BaseDataset]: Dataset type.
        """

        raise NotImplementedError

    def get_transform(self, **kwargs) -> BaseTransform:
        """
        Returns transform, which preprocesses data.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete datamodule.

        Returns:
            BaseTransform: Transform.
        """

        raise NotImplementedError

    def get_train_val_test_split(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Returns train, validation, and test datasets.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete datamodule.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: Train, validation, test dataset.
        """

        raise NotImplementedError

    def prepare_data(self) -> None:
        """Downloads datasets."""

        # download
        self.dataset_type(self.data_dir, download=True)

    def instantiate_dataset(self, **kwargs) -> BaseDataset:
        """
        Returns instance of dataset, where arguments 'root', 'download', 'transform' of BaseDataset
        class do not have to be passed.

        Returns:
            BaseDataset: Dataset instance.
        """

        return self.dataset_type(
            root=self.data_dir, download=False, transform=self.transform, **kwargs
        )

    def setup(self, stage: str) -> None:
        """
        Loads datasets.

        Args:
            stage (str): Mode (fit/test/predict).
        """

        if self.train_dataset is None and self.val_dataset is None and self.test_dataset is None:
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = self.get_train_val_test_split()

    def train_dataloader(self) -> DataLoader:
        """
        Returns dataloader for training.

        Returns:
            DataLoader: Dataloader.
        """

        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collator,  # type: ignore
            num_workers=self.num_workers,
            pin_memory=self.num_gpus > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns dataloader for validation.

        Returns:
            DataLoader: Dataloader.
        """

        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,  # type: ignore
            num_workers=self.num_workers,
            pin_memory=self.num_gpus > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns dataloader for testing.

        Returns:
            DataLoader: Dataloader.
        """

        return DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,  # type: ignore
            num_workers=self.num_workers,
            pin_memory=self.num_gpus > 0,
        )
