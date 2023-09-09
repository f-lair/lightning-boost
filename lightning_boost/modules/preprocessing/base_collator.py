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


from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import get_worker_info


class BaseCollator:
    def __init__(
        self, pad_val: int = 0, pad_shape: List[int] = [], pad_dims: List[int] = []
    ) -> None:
        """
        Initializes collator, which transforms a list of batch items to a tensor with additional
        batch dimension. In case of different shapes, padding is applied.

        Args:
            pad_val (int, optional): Padding value. Defaults to 0.
            pad_shape (List[int], optional): Dimension sizes after padding. Must match pad_dims. Defaults to [].
            pad_dims (List[int], optional): Dimensions in which padding is applied. Defaults to [].
        """

        assert len(pad_shape) == len(
            pad_dims
        ), "Number of padded dims must match dimensionality of padded shape!"

        self.pad_val = pad_val
        self.pad_shape = pad_shape
        self.pad_dims = pad_dims
        self.collate_fn = self.get_collate_fn()

    def get_collate_fn(self) -> Dict[str, Callable[[List[Tensor]], Tensor]]:
        """
        Returns collator functions for each data type in inputs and targets.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete collator.

        Returns:
            Dict[str, Callable[[List[Tensor]], Tensor]]: Collator functions that take a list of tensors and return a single tensor.
        """

        raise NotImplementedError

    def __call__(
        self, batch: List[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Performs collation for inputs and targets of a batch.

        Args:
            batch (List[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]): List of (Inputs, Targets) pairs.

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]: Collated inputs and targets.
        """

        return tuple(self.collate_dict(idx, item, batch) for idx, item in enumerate(batch[0]))

    def collate_dict(
        self,
        item_idx: int,
        item: Dict[str, Tensor],
        batch: List[Tuple[Dict[str, Tensor], Dict[str, Tensor]]],
    ) -> Dict[str, Tensor]:
        """
        Performs collation of batch data for a single data item, i.e., either input or target data.

        Args:
            item_idx (int): Index of the data item.
            item (Dict[str, Tensor]): Data item, containing several tensors.
            batch (List[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]): Full batch data: Each batch item consists of input and target data, which in turn contain several tensors.

        Returns:
            Dict[str, Tensor]: Collated batch data for the given data item, i.e., it contains several tensors, which include a batch dimension.
        """

        return {
            key: self.collate_fn[key]([sample[item_idx][key] for sample in batch]) for key in item
        }

    def pad_collate_nd(self, batch: List[Tensor]) -> Tensor:
        """
        Pads and concatenates a list of possibly differently shaped n-dimensional tensors along a
        new batch dimension to a single tensor.

        Args:
            batch (List[Tensor]): List of possibly differently shaped n-dimensional tensors.

        Returns:
            Tensor: Padded and concatenated tensors.
        """

        B = len(batch)
        n_dims = batch[0].dim()
        dims = [max([sample.size(dim) for sample in batch]) for dim in range(n_dims)]
        for idx, pad_dim in enumerate(self.pad_dims):
            dims[pad_dim] = max(dims[pad_dim], self.pad_shape[idx])

        # optimization using shared memory to avoid a copy when passing data to the main process
        if get_worker_info() is not None:
            elem = batch[0]
            numel = int(
                B * torch.tensor(dims).prod().item()  # (elem.element_size() // elem.numel())
            )
            storage = elem._typed_storage()._new_shared(numel, device=elem.device)
            out_shm = elem.new(storage)
            out_shm[:] = self.pad_val
            out = out_shm.view(B, *dims)
        else:
            out = torch.full((B, *dims), self.pad_val, dtype=batch[0].dtype)  # type: ignore

        for idx, sample in enumerate(batch):
            inplace_slice = (idx,) + tuple(slice(sample_dim) for sample_dim in sample.size())
            insert_slice = (slice(None),) * n_dims
            out[inplace_slice] = sample[insert_slice]

        return out

    def flatten_collate(self, batch: List[Tensor]) -> Tensor:
        """
        Concatenates a list of 0-dimensional (scalar) tensors along a new dimension to a single
        1-dimensional tensor.

        Args:
            batch (List[Tensor]): List of 0-dimensional tensors.

        Returns:
            Tensor: Concatenated 1-dimensional tensor.
        """

        # optimization using shared memory to avoid a copy when passing data to the main process
        out = None
        if get_worker_info() is not None:
            elem = batch[0]
            numel = len(batch)  # (elem.element_size() // elem.numel()))
            storage = elem._typed_storage()._new_shared(numel, device=elem.device)
            out_shm = elem.new(storage)
            out = out_shm.view(len(batch))
        return torch.stack(batch, dim=0, out=out)
