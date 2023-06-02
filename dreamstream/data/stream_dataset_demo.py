from collections import namedtuple
import itertools
import logging

from typing import List, Callable, Tuple
from string import ascii_letters

import random
import numpy as np

import torch

from torch.utils.data import IterableDataset, DataLoader


LOGGER = logging.getLogger(__file__)


def partition_by_sum(values: List[float], num_partitions: int, shuffle: bool = False, sort: bool = True):
    assert not shuffle or not sort, "shuffle and sort cannot both be True"

    num_values = len(values)

    if sort:
        indices = np.argsort(values)
    else:
        indices = list(range(num_values))

    if shuffle:
        random.shuffle(indices)

    partitions = [[] for _ in range(num_partitions)]
    indices_partitions = [[] for _ in range(num_partitions)]
    sums = [0] * num_partitions
    for i in indices:
        min_sum_idx = sums.index(min(sums))
        partitions[min_sum_idx].append(values[i])
        indices_partitions[min_sum_idx].append(i)
        sums[min_sum_idx] += values[i]

    return partitions, indices_partitions


Metadata = namedtuple("Metadata", ["name"])


class MyIterableDataset(IterableDataset):
    def __init__(
        self, data_list: list, batch_size: int, shuffle: bool = False, drop_last: bool = False, name: str = None
    ):
        self.data_list = data_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.name = name

    def process_data(self, data):
        for x in data:
            worker = torch.utils.data.get_worker_info()
            worker_id = worker.id if worker is not None else -1
            worker_seed = worker.seed if worker is not None else -1

            yield x, self.name, worker_id, worker_seed

    def get_stream(self, data_list):
        return itertools.chain.from_iterable(map(self.process_data, data_list))

    def get_streams(self):
        """Create a stream of data from the dataset.

        Each stream element is a batch from the dataset (a list of `batch_size` elements), where each element is a
        tuple of the form `(data, name, worker_id, worker_seed)`.

        Each batch is sampled without
        """
        if self.shuffle:
            data_list = random.sample(self.data_list, len(self.data_list))
        else:
            data_list = self.data_list

        # partition naively
        partitioned_shuffled_data_list = [data_list[i :: self.batch_size] for i in range(self.batch_size)]

        data_streams = [self.get_stream(partitioned_shuffled_data_list[i]) for i in range(self.batch_size)]
        if self.drop_last:
            data_stream = zip(*data_streams)
        else:
            data_stream = itertools.zip_longest(*data_streams)
            data_stream = (list(filter(lambda x: x is not None, batch_parts)) for batch_parts in data_stream)

        return data_stream

    def __iter__(self):
        return self.get_streams()

    @staticmethod
    def custom_collate(batch: List[Tuple[torch.Tensor, str, int, int]]):
        batch = list(zip(*batch))
        metadata = Metadata(batch[1])
        data = batch[0]
        return data, metadata

    @classmethod
    def split_datasets(cls, data_list, batch_size, max_workers):
        """Split the dataset into multiple datasets, each with a subset of the data."""
        if max_workers > batch_size:
            max_workers = batch_size
            LOGGER.warning(f"max_workers cannot be greater than batch_size. Setting max_workers to {max_workers}")

        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                max_workers = n
                LOGGER.warning(f"`max_workers` must be a divisor of batch_size. Setting max_workers to {max_workers}.")
                break

        # split data_list among max_workers such that each worker gets approximately the same total data length
        print(batch_size, max_workers)
        data_lengths = [len(x) for x in data_list]
        _, indices = partition_by_sum(data_lengths, max_workers)
        partitioned_data = [[data_list[i] for i in ids] for ids in indices]

        batch_size_per_worker = batch_size // max_workers
        names = ascii_letters.upper()

        datasets = [
            cls(data_list=partitioned_data[i], batch_size=batch_size_per_worker, name=names[i])
            for i in range(max_workers)
        ]
        print(len(datasets))
        return datasets


class MultiStreamDataLoader:
    def __init__(
        self,
        datasets: List[IterableDataset],
        multiprocessed: bool = True,
        drop_last: bool = False,
        collate_fn: Callable = None,
    ) -> None:
        """Create a dataloader that iterates over multiple datasets in parallel.
        The number of workers is equal to the number of datasets.

        Args:
            datasets (List[IterableDataset]): _description_
            multiprocessed (bool, optional): _description_. Defaults to True.
            drop_last (bool, optional): _description_. Defaults to False.
        """
        self.datasets = datasets
        self.multiprocessed = multiprocessed
        self.drop_last = drop_last

        self.num_workers = len(datasets) if multiprocessed else 0

        if collate_fn is None:
            collate_fn = torch.utils.data._utils.collate.default_collate

        self.collate_fn = collate_fn

    def get_stream_loaders(self):
        num_workers = 1 if self.multiprocessed else 0
        stream_loaders = [DataLoader(dataset, num_workers=num_workers, batch_size=None) for dataset in self.datasets]

        if self.drop_last:
            stream_loaders = zip(*stream_loaders)
        else:
            stream_loaders = itertools.zip_longest(*stream_loaders)
            stream_loaders = (filter(lambda x: x is not None, batch_parts) for batch_parts in stream_loaders)

        return stream_loaders

    def _collate(self):
        pass

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            # TODO Collate batch parts to batches here
            batch_parts_flat = list(itertools.chain.from_iterable(batch_parts))
            collated_batch = self.collate_fn(batch_parts_flat)
            yield collated_batch


if __name__ == "__main__":
    # List of examples consisting of "chunks" of data from a single file (e.g. a1 through a7 is 7 chunks from file a)
    data_list = [
        ["a1", "a2", "a3", "a4", "a5", "a6", "a7"],
        ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"],
        ["c1", "c2", "c3", "c4"],
        ["d1", "d2", "d3", "d4", "d5", "d6"],
        ["e1", "e2", "e3", "e4", "e5"],
        ["f1", "f2", "f3", "f4", "f5", "f6"],
        ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"],
        ["h1", "h2", "h3", "h4", "h5", "h6", "h7"],
        ["i1", "i2", "i3"],
        ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9"],
    ]

    datasets = MyIterableDataset.split_datasets(data_list, batch_size=6, max_workers=3)
    loader = MultiStreamDataLoader(
        datasets, multiprocessed=True, drop_last=False, collate_fn=MyIterableDataset.custom_collate
    )

    for batch, metadata in loader:
        print(batch, metadata)
