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


import math
from typing import Tuple, Type

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, random_split

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
        num_folds: int = 1,
        fold_index: int = 0,
        fold_seed: int = 42,
        **kwargs,
    ):
        """
        Initializes datamodule.

        Args:
            data_dir (str, optional): Directory, where dataset is stored. Defaults to './data/download/'.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of additional workers used for data loading. Defaults to 0.
            num_gpus (int, optional): Number of used GPUs. Defaults to 0.
            shuffle (bool, optional): Whether training dataset is shuffled. Defaults to True.
            val_ratio (float, optional): Ratio of validation split. Defaults to 0.1.
            test_ratio (float, optional): Ratio of validation split. Defaults to 0.1.
            num_folds (int, optional): Number of folds in cross-validation. Defaults to 1.
            fold_index (int, optional): Fold index in cross-validation. Defaults to 0.
            fold_seed (int, optional): RNG seed for fold generation. Defaults to 42.
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
        assert 0 <= fold_index < num_folds, "Fold index must be in [0, num_folds)!"

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.shuffle = shuffle
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_folds = num_folds
        self.fold_index = fold_index
        self.fold_seed = fold_seed

        self.collator = self.get_collator(**kwargs)
        self.dataset_type = self.get_dataset_type(**kwargs)
        self.transform = self.get_transform(**kwargs)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.cv_indices = None
        self.fold_len = None

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

    def get_train_test_split(self) -> Tuple[Dataset, Dataset]:
        """
        Returns train (incl. validation data) and test datasets.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete datamodule.

        Returns:
            Tuple[Dataset, Dataset]: Train, test dataset.
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
            stage (str): Mode (fit/validate/test).
        """

        if self.train_dataset is None and self.val_dataset is None and self.test_dataset is None:
            self.train_dataset, self.test_dataset = self.get_train_test_split()

    def get_train_val_split(self) -> Tuple[Dataset, Dataset]:
        """
        Returns reduced train and validation datasets from whole train dataset.
        Applies basic random split.

        Returns:
            Tuple[Dataset, Dataset]: Train, validation dataset.
        """

        train_dataset, val_dataset = random_split(
            self.train_dataset, [1 - self.val_ratio, self.val_ratio]  # type: ignore
        )

        return train_dataset, val_dataset

    def get_cv_train_val_split(self) -> Tuple[Dataset, Dataset]:
        """
        Returns reduced train and validation datasets from whole train dataset.
        Applies basic k-fold cross-validation split.

        Returns:
            Tuple[Dataset, Dataset]: Train, validation dataset.
        """

        assert self.fold_len is not None
        assert self.cv_indices is not None

        slice_1 = slice(0, self.fold_index * self.fold_len)
        slice_2 = slice((self.fold_index + 1) * self.fold_len, len(self.train_dataset))  # type: ignore
        train_indices = torch.concat((self.cv_indices[slice_1], self.cv_indices[slice_2]))  # type: ignore
        val_indices = self.cv_indices[  # type: ignore
            self.fold_index * self.fold_len : (self.fold_index + 1) * self.fold_len
        ]

        train_dataset = Subset(self.train_dataset, train_indices)  # type: ignore
        val_dataset = Subset(self.train_dataset, val_indices)  # type: ignore

        return train_dataset, val_dataset

    def determine_cv_indices(self) -> None:
        """
        Determines data index permutation for cross-validation.
        """

        assert self.train_dataset is not None

        if self.cv_indices is None:
            rng = torch.Generator()
            rng.manual_seed(self.fold_seed)
            self.cv_indices = torch.randperm(len(self.train_dataset), dtype=torch.int64, generator=rng)  # type: ignore

    def determine_fold_len(self) -> None:
        """
        Determines length of each fold in cross-validation.
        """

        assert self.train_dataset is not None

        if self.fold_len is None:
            self.fold_len = int(math.ceil(len(self.train_dataset) / self.num_folds))  # type: ignore

    def _get_train_val_split(self) -> Tuple[Dataset, Dataset]:
        """
        Returns reduced train and validation datasets from whole train dataset.
        Applies regular split or cross-validation, depending on number of folds.

        Returns:
            Tuple[Dataset, Dataset]: Train, validation dataset.
        """

        if self.num_folds > 1:
            self.determine_cv_indices()
            self.determine_fold_len()
            train_dataset, val_dataset = self.get_cv_train_val_split()
        else:
            train_dataset, val_dataset = self.get_train_val_split()

        return train_dataset, val_dataset

    def train_dataloader(self) -> DataLoader:
        """
        Returns dataloader for training.

        Returns:
            DataLoader: Dataloader.
        """

        train_dataset, _ = self._get_train_val_split()

        return DataLoader(
            train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collator,  # type: ignore
            num_workers=self.num_workers,
            pin_memory=self.num_gpus > 0,
        )

    def val_dataloader(self, val_dataset: Dataset | None = None) -> DataLoader:
        """
        Returns dataloader for validation.

        Returns:
            DataLoader: Dataloader.
        """

        _, val_dataset = self._get_train_val_split()

        return DataLoader(
            val_dataset,  # type: ignore
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
