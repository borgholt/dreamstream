import itertools
import logging
import math
import pathlib
import time

from typing import Dict, List, Callable, Optional, Generator, Union
from tqdm import tqdm

import random
import numpy as np
import pandas as pd

import torch
import torchaudio

from torch.utils.data import IterableDataset, DataLoader

from dreamstream.data_objects import AudioSample, StreamState
from dreamstream.stream_tensor import StreamTensor


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


class AudioStreamDataset(IterableDataset):
    def __init__(
        self,
        file_list: List[str],
        chunk_seconds: float = 1.0,
        transform: Callable = None,
        file_metadata: Dict[str, torchaudio.backend.common.AudioMetaData] = None,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        """Create a dataset to stream audio files from a file_list.

        On iteration, the dataset reads an audio file from a file list, applies a transform, and chunks the audio
        into chunks of length `chunk_seconds`. Each chunk is returned as a `AudioSample` instance.

        If `batch_size > 1`, the dataset returns a batch of `batch_size` audio chunks from different files on iteration.
        If `shuffle=True`, the dataset shuffles the file list to give different batches on each iteration.
        If `drop_last=True`, the dataset drops the last batch if it is smaller than `batch_size`.
        These attributes/behaviors are useful when serving the dataset using a `MultiStreamDataLoader`.

        Files are read only once before chunking and iteration.

        Args:
            file_list (List[str]]): A list of audio file paths.
            chunk_seconds (float, optional): The length of each chunk in seconds. Defaults to 1.0.
            transform (Callable, optional): A function to apply to the audio before chunking. Defaults to None.
            file_metadata (Dict[str, torchaudio.backend.common.AudioMetaData], optional): A dictionary mapping file
                of audio file paths to their metadata. Defaults to None.
            batch_size (Optional[int], optional): The number of audio chunks to return per iteration. Defaults to None.
            shuffle (bool, optional): If `batch_size > 1`, shuffle the file list before defining streams for each
                batch element. This gives different batches on each iteration. Defaults to False.
            drop_last (bool, optional): If `batch_size > 1`, drop the last batch if it is smaller than `batch_size`.
        """
        self.file_list = file_list
        self.chunk_seconds = chunk_seconds
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.file_metadata = file_metadata or self.get_file_metadata()
        self.file_lengths = {file: m.num_frames / m.sample_rate for file, m in self.file_metadata.items()}

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise AttributeError("batch_size must be set before accessing it")
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        assert batch_size is None or batch_size > 0, "batch_size must be None or greater than 0"
        self._batch_size = batch_size

    def get_file_metadata(self) -> Generator[float, None, None]:
        iterator = tqdm(self.file_list, desc="Getting file metadata", unit="file")
        return {file: torchaudio.info(file) for file in iterator}

    def process_data(self, file):
        # Load file
        file_metadata = self.file_metadata[file]
        audio, sample_rate = torchaudio.load(file)
        num_frames = audio.shape[1]

        # Apply transforms
        if self.transform:
            audio = self.transform(audio)

        # Chunk transformed audio
        stride = audio.shape[1] / num_frames
        chunk_size = round(self.chunk_seconds * sample_rate / stride)
        audio_chunks = [audio[:, i : i + chunk_size] for i in range(0, num_frames, chunk_size)]

        for i, audio_chunk in enumerate(audio_chunks):
            audio_sample = AudioSample(
                data=audio,
                is_first=(i == 0),
                is_last=(i == len(audio_chunks) - 1),
                length=audio_chunk.shape[1],
                chunk_index=i,
                num_chunks=len(audio_chunks),
                file=file,
                file_metadata=file_metadata,
                id=file,
            )
            yield audio_sample

    def get_stream(self, file_list):
        return itertools.chain.from_iterable(map(self.process_data, file_list))

    def get_streams(self):
        """Create a stream of data from the dataset.

        Each stream element is a batch from the dataset (a list of `batch_size` elements), where each element is a
        tuple of the form `(data, name, worker_id, worker_seed)`.

        Each batch is sampled without
        """
        if self.shuffle:
            file_list = random.sample(self.file_list, len(self.file_list))
        else:
            file_list = self.file_list

        # partition file list into batch_size file lists
        data_streams = [self.get_stream(file_list[i :: self.batch_size]) for i in range(self.batch_size)]
        if self.drop_last:
            data_stream = zip(*data_streams)
        else:
            data_stream = itertools.zip_longest(*data_streams)
            data_stream = (list(filter(lambda x: x is not None, batch_parts)) for batch_parts in data_stream)

        return data_stream

    def __iter__(self):
        return self.get_streams()

    @staticmethod
    def custom_collate(batch: List[AudioSample]):
        # sort by length
        batch = sorted(batch, key=lambda x: x.length, reverse=True)
        max_len = max([sample.data.shape[1] for sample in batch])

        # collate batch data tensor        
        data = torch.zeros(len(batch), batch[0].data.shape[0], max_len)
        for i, sample in enumerate(batch):
            data[i, :, : sample.data.shape[1]] = sample.data
        
        # collate batch metadata
        is_first = torch.tensor([sample.is_first for sample in batch])
        is_last = torch.tensor([sample.is_last for sample in batch])
        lengths = torch.tensor([sample.length for sample in batch])
        chunk_index = torch.tensor([sample.chunk_index for sample in batch])
        num_chunks = torch.tensor([sample.num_chunks for sample in batch])
        ids = [sample.id for sample in batch]

        # create stream tensor and its state
        stream_state = StreamState(ids, is_first, is_last, lengths, chunk_index, num_chunks)
        stream_tensor = StreamTensor(data, stream_state=stream_state)
        return stream_tensor

    def split_dataset(self, batch_size: int, max_workers: int):
        """Split the dataset into multiple datasets, each with a subset of the data."""
        assert max_workers > 0, "max_workers must be greater than 0"

        if max_workers > batch_size:
            max_workers = batch_size
            LOGGER.warning(f"max_workers cannot be greater than batch_size. Setting max_workers to {max_workers}")

        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                max_workers = n
                LOGGER.warning(f"`max_workers` must be a divisor of batch_size. Setting max_workers to {max_workers}.")
                break

        # split file_list among max_workers such that each worker gets approximately the same total data length
        print(batch_size, max_workers)
        file_lengths = [self.file_lengths[file] for file in self.file_list]
        num_chunks_per_file = [round(l / self.chunk_seconds) for l in file_lengths]
        partitioned_num_chunks, indices = partition_by_sum(num_chunks_per_file, max_workers)
        partitioned_data = [[self.file_list[i] for i in ids] for ids in indices]
        partitioned_file_metadata = [
            {file: self.file_metadata[file] for file in partition} for partition in partitioned_data
        ]

        batch_size_per_worker = batch_size // max_workers

        datasets = [
            self.__class__(
                file_list=partitioned_data[i],
                file_metadata=partitioned_file_metadata[i],
                chunk_seconds=self.chunk_seconds,
                batch_size=batch_size_per_worker,
                transform=self.transform,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
            )
            for i in range(max_workers)
        ]
        print(len(datasets))
        return datasets


class MultiStreamDataLoader:
    def __init__(
        self,
        dataset: Union[IterableDataset, List[IterableDataset]],
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Callable = None,
    ) -> None:
        """Create a dataloader that iterates over multiple dataset in parallel.

        If a list of datasets is given, each one is given to a worker to process in parallel.
        If a single dataset is given, it is split into `num_workers` dataset, each of which is given to a worker to
        process in parallel. We assume the dataset has a `batch_size` attribute, and a `split_dataset` method that,
        given a `batch_size` and `max_workers`, splits the dataset into multiple dataset.

        Args:
            dataset (Union[IterableDataset, List[IterableDataset]]): A single or a list of IterableDataset instances.
            num_workers (int, optional): When a single dataset is given, it is split into `num_workers` subsets and
                each to be processed in parallel by a single worker. When a list of datasets is given, `num_workers`
                has no effect. Defaults to 0.
            drop_last (bool, optional): Drop the last batch(es) if smaller than `batch_size`. Defaults to False.
            collate_fn (Callable, optional): A function that takes a list of batch parts and collates them into a
                batch. Defaults to None.
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last

        if isinstance(dataset, list):
            self.num_workers = len(dataset)

        if collate_fn is None:
            collate_fn = torch.utils.data._utils.collate.default_collate

        self.collate_fn = collate_fn

    def get_stream_loaders(self):
        if isinstance(self.dataset, IterableDataset):
            num_workers = max(1, self.num_workers)
            datasets = self.dataset.split_dataset(batch_size=self.batch_size, max_workers=num_workers)
        else:
            datasets = self.dataset

        num_workers = 0 if self.num_workers == 0 else 1
        stream_loaders = [DataLoader(dataset, num_workers=num_workers, batch_size=None) for dataset in datasets]

        if self.drop_last:
            stream_loaders = zip(*stream_loaders)
        else:
            stream_loaders = itertools.zip_longest(*stream_loaders)
            stream_loaders = (filter(lambda x: x is not None, batch_parts) for batch_parts in stream_loaders)

        return stream_loaders

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            # TODO Collate batch parts to batches here
            batch_parts_flat = list(itertools.chain.from_iterable(batch_parts))
            
            filenames = [batch_part.file.split("/")[-1].split(".")[0] for batch_part in batch_parts_flat]
            chunk_idx = [batch_part.chunk_index for batch_part in batch_parts_flat]
            is_first = [int(batch_part.is_first) for batch_part in batch_parts_flat]
            is_last = [int(batch_part.is_last) for batch_part in batch_parts_flat]
            
            filenames_with_index = [f"{filename} {idx:2d}" for filename, idx in zip(filenames, chunk_idx)]
            print(filenames_with_index, is_first, is_last)
            
            collated_batch = self.collate_fn(batch_parts_flat)
            yield collated_batch


if __name__ == "__main__":
    source_file = pathlib.Path("/m2/research/source/wsj/test_dev93.txt")
    source_df = pd.read_csv(source_file)
    source_df.filename = source_df.filename.apply(lambda x: x + ".wav")

    datasets = AudioStreamDataset(file_list=source_df.filename, chunk_seconds=1.0, shuffle=False)
    loader = MultiStreamDataLoader(
        datasets,
        batch_size=256,
        num_workers=8,
        shuffle=False,
        drop_last=False,
        collate_fn=AudioStreamDataset.custom_collate,
    )

    files_seen = set()
    samples_seen = []
    ts = time.time()
    for batch in loader:
        files_seen.update(batch.stream_state.ids)

        sample_ids = [f"{batch.stream_state.ids[i]} {batch.stream_state.chunk_index[i]:2d}" for i in range(len(batch.stream_state.ids))]
        samples_seen.extend(sample_ids)
        
    print(f"Time taken: {time.time() - ts:.2f} s")
    
    print(f"Expected files: {len(source_df)}")
    print(f"Expected samples: ", sum([math.ceil(l / datasets.chunk_seconds) for l in datasets.file_lengths.values()]))
    print(f"Files seen: {len(files_seen)}")
    print(f"Samples seen: {len(samples_seen)}")
    print(f"Unique samples seen: {len(set(samples_seen))}")

    
    import IPython
    IPython.embed(using=False)
