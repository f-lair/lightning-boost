from typing import Sequence, Tuple, Type

from torch.utils.data import Dataset, Subset

from lightning_boost.data.datamodules import BaseDatamodule
from tests.dummy.data.datasets import DummyDataset
from tests.dummy.modules.preprocessing import DummyCollator, DummyTransform


class DummyDatamodule(BaseDatamodule):
    def __init__(
        self,
        data_dir: str = './data/download/',
        batch_size: int = 1,
        num_workers: int = 0,
        num_gpus: int = 0,
        shuffle: bool = False,
        val_ratio: float = 0.25,
        test_ratio: float = 0.25,
        dataset_size: Sequence[int] = (8, 42),
        **kwargs
    ):
        self.dataset_size = dataset_size
        super().__init__(
            data_dir, batch_size, num_workers, num_gpus, shuffle, val_ratio, test_ratio, **kwargs
        )

    def get_collator(self, **kwargs) -> DummyCollator:
        return DummyCollator()

    def get_dataset_type(self) -> Type[DummyDataset]:
        return DummyDataset

    def get_transform(self, **kwargs) -> DummyTransform:
        return DummyTransform()

    def get_train_val_test_split(self) -> Tuple[Dataset, Dataset, Dataset]:
        dataset = self.instantiate_dataset(size=self.dataset_size)
        train_ratio = 1.0 - (self.val_ratio + self.test_ratio)

        train_range = (0, int(train_ratio * len(dataset)))
        val_range = (train_range[1], int(train_range[1] + self.val_ratio * len(dataset)))
        test_range = (val_range[1], int(val_range[1] + self.test_ratio * len(dataset)))

        train_dataset = Subset(dataset, list(range(train_range[0], train_range[1])))
        val_dataset = Subset(dataset, list(range(val_range[0], val_range[1])))
        test_dataset = Subset(dataset, list(range(test_range[0], test_range[1])))

        return train_dataset, val_dataset, test_dataset
