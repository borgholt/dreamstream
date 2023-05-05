import itertools
import logging
import math
import pathlib
import time

from typing import Dict, List, Callable, Optional, Generator, Tuple, Union
from tqdm import tqdm

import random
import numpy as np
import pandas as pd

import torch
import torchaudio

from torch.utils.data import IterableDataset, DataLoader

from dreamstream.data_objects import AudioSample
from dreamstream.stream_tensor import StreamTensor, StreamState


LOGGER = logging.getLogger(__file__)


def partition_by_sum(
    values: List[float], num_partitions: int, shuffle: bool = False, sort: bool = True
) -> Tuple[List[List[float]], List[List[int]]]:
    """Partition a list of values into `num_partitions` partitions such that the sum of values in each partition is
    approximately equal.

    Args:
        values (List[float]): A list of values to partition.
        num_partitions (int): The number of partitions to create.
        shuffle (bool, optional): If True, shuffle the values before partitioning. Defaults to False.
        sort (bool, optional): If True, sort the values ascending before partitioning. This can help create a more exact
            partitioning if shuffling is not needed. Defaults to True.

    Returns:
        Tuple[List[List[float]], List[List[int]]]: A tuple of lists of partitions and indices. The partitions are lists
            of values, and the indices are lists of indices into the original list of values.
    """
    if shuffle and sort:
        raise ValueError("`shuffle` and `sort` must not both be True.")

    if sort:
        indices = torch.argsort(values) if isinstance(values, torch.Tensor) else np.argsort(values)
    else:
        indices = list(range(len(values)))

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


def force_is_last_on_last_batch(
    stream: Generator[List[List[AudioSample]], None, None]
) -> Generator[List[List[AudioSample]], None, None]:
    # yield all but last batch
    to_yield = next(stream)
    for value in stream:
        yield to_yield
        to_yield = value

    # yield last batch with is_last=True on all samples
    for worker_outputs in to_yield:
        for sample in worker_outputs:
            sample.is_last = True

    yield to_yield


def concatenate_ragged(
    batch: List[torch.Tensor], pad_value: float = 0, dim: int = -1
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Zero pad batch of tensors of some shape (*, dim, *) to maximum temporal length and concatenate"""
    sequence_lengths = [tensor.shape[dim] for tensor in batch]

    N = len(batch)
    T = max(sequence_lengths)

    # get shape of batch with full temporal dimension
    collated_shape = list(batch[0].shape)
    collated_shape[dim] = T
    collated_shape = [N] + collated_shape  # (B, *, D, *)

    # move padding dimension to end to allow easy indexing into collated_batch below
    collated_shape[dim], collated_shape[-1] = collated_shape[-1], collated_shape[dim]  # (B, *, D)

    dtype = batch[0].dtype
    collated_batch = torch.full(collated_shape, dtype=dtype, fill_value=pad_value)  # (B, *, D)
    for i, seq_len in enumerate(sequence_lengths):
        collated_batch[i, ..., :seq_len] = batch[i].transpose(dim, -1)

    # revert transpose
    collated_batch = collated_batch.transpose(dim, -1)  # (B, *, D, *)

    return collated_batch, torch.LongTensor(sequence_lengths)


class AudioStreamDataset(IterableDataset):
    def __init__(
        self,
        file_list: List[str],
        chunk_seconds: float = 1.0,
        transform: Callable = None,
        file_metadata: Dict[str, torchaudio.backend.common.AudioMetaData] = None,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = False,
        drop_last: Optional[bool] = False,
    ):
        """Create a dataset to stream audio files from a file_list.

        On iteration, the dataset reads an audio file from a file list, applies a transform, and chunks the audio
        into chunks of length `chunk_seconds`. Each chunk is returned as a `AudioSample` instance.

        If `batch_size > 1`, the dataset returns a batch of `batch_size` audio chunks from different files on iteration.
        If `shuffle=True`, the dataset shuffles the file list to give different batches on each iteration.
        If `drop_last=True`, the dataset drops the last batch if it is smaller than `batch_size`.
        These attributes/behaviors are useful when serving the dataset using a `MultiStreamDataLoader` but can be
        ignored if the dataset is used standalone.

        Files are read only once before chunking and iteration.

        Args:
            file_list (List[str]]): A list of audio file paths.
            chunk_seconds (float, optional): The length of each chunk in seconds. Defaults to 1.0.
            transform (Callable, optional): A function to apply to the audio before chunking. Defaults to None.
            file_metadata (Dict[str, torchaudio.backend.common.AudioMetaData], optional): A dictionary mapping file
                of audio file paths to their metadata. Defaults to None.
            batch_size (Optional[int], optional): The number of audio chunks to return per iteration. Defaults to 1.
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
        iterator = tqdm(self.file_list, desc="Getting file metadata", unit="file", delay=5)
        return {file: torchaudio.info(file) for file in iterator}

    def process_data(self, file: str) -> Generator[AudioSample, None, None]:
        # TODO (JDH): For vseq, we can generalize this to support several modalities with Loader classes by adding
        #             a Chunker class per modality that splits a loaded file into chunks.
        # Load file
        file_metadata = self.file_metadata[file]
        audio, sample_rate = torchaudio.load(file)
        num_frames = audio.shape[-1]

        # Apply transforms
        if self.transform:
            audio = self.transform(audio)

        # Chunk transformed audio
        stride = num_frames / audio.shape[-1]
        chunk_size = round(self.chunk_seconds * sample_rate / stride)
        audio_chunks = [audio[..., i : i + chunk_size] for i in range(0, audio.shape[-1], chunk_size)]

        for i, audio_chunk in enumerate(audio_chunks):
            audio_sample = AudioSample(
                data=audio,
                is_first=(i == 0),
                is_last=(i == len(audio_chunks) - 1),
                length=audio_chunk.shape[-1],
                chunk_index=i,
                num_chunks=len(audio_chunks),
                file=file,
                file_metadata=file_metadata,
                id=file,
            )
            yield audio_sample

    def get_stream(self, file_list: List[str]) -> Generator[AudioSample, None, None]:
        return itertools.chain.from_iterable(map(self.process_data, file_list))

    def get_streams(self) -> Generator[List[AudioSample], None, None]:
        """Create a stream of data from the dataset.

        The stream is a generator that yields batches of data. Each batch is a list of `batch_size` elements, and each
        element is an AudioSample.

        Each batch is sampled to consist only of unique files. This is done by partitioning the file list into
        `batch_size` file lists, and creating a stream for each file list. If `shuffle=True`, the file list is
        shuffled before partitioning.
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

    def __iter__(self) -> Generator[List[AudioSample], None, None]:
        return self.get_streams()

    @staticmethod
    def custom_collate(batch: List[AudioSample]) -> StreamTensor:
        """Collate a batch of AudioSample instances into a StreamTensor which is a torch.Tensor wth a StreamState."""
        # sort by length
        batch = sorted(batch, key=lambda x: x.length, reverse=True)

        data = [sample.data for sample in batch]
        data, lengths = concatenate_ragged(data, dim=-1)

        # collate batch metadata
        is_first = torch.tensor([sample.is_first for sample in batch])
        is_last = torch.tensor([sample.is_last for sample in batch])
        chunk_index = torch.tensor([sample.chunk_index for sample in batch])
        num_chunks = torch.tensor([sample.num_chunks for sample in batch])
        ids = [sample.id for sample in batch]

        # create stream tensor and its state
        stream_state = StreamState(ids, is_first, is_last, lengths, chunk_index, num_chunks)
        stream_tensor = StreamTensor(data, stream_state=stream_state)
        return stream_tensor

    def split(self, max_workers: int, batch_size: int, shuffle: bool, drop_last: bool):
        """Split the dataset into multiple datasets, each with a unique of the data."""
        assert max_workers > 0, "max_workers must be greater than 0"

        if max_workers > batch_size:
            max_workers = batch_size
            LOGGER.warning(f"max_workers cannot be greater than batch_size. Setting max_workers to {max_workers}")

        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                max_workers = n
                LOGGER.warning(f"`max_workers` must be a divisor of batch_size. Setting max_workers to {max_workers}.")
                break

        # split file_list among max_workers such that each worker gets approximately the same total number of chunks.
        file_lengths = [self.file_lengths[file] for file in self.file_list]
        num_chunks_per_file = [round(l / self.chunk_seconds) for l in file_lengths]
        partitioned_num_chunks, indices = partition_by_sum(num_chunks_per_file, max_workers)
        file_lists = [[self.file_list[i] for i in ids] for ids in indices]
        file_metadatas = [{file: self.file_metadata[file] for file in partition} for partition in file_lists]

        batch_size_per_worker = batch_size // max_workers

        datasets = [
            self.__class__(
                file_list=file_lists[i],
                file_metadata=file_metadatas[i],
                chunk_seconds=self.chunk_seconds,
                batch_size=batch_size_per_worker,
                transform=self.transform,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            for i in range(max_workers)
        ]
        return datasets


class MultiStreamDataLoader:
    def __init__(
        self,
        dataset: Union[IterableDataset, List[IterableDataset]],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: Optional[torch.device] = "",
    ) -> None:
        """Create a dataloader that iterates over multiple dataset in parallel.

        If a list of datasets is given, each one is given to a worker to process in parallel.
        If a single dataset is given, it is split into `num_workers` dataset, each of which is given to a worker to
        process in parallel. We assume the dataset has a `batch_size` attribute, and a `split` method that,
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
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.worker_init_fn = worker_init_fn

        if isinstance(dataset, list):
            self.num_workers = len(dataset)

        if collate_fn is None:
            collate_fn = torch.utils.data._utils.collate.default_collate

        self.collate_fn = collate_fn

        self.dataloader_kwargs = dict(
            batch_size=None,
            num_workers=(0 if self.num_workers == 0 else 1),
            pin_memory=pin_memory,
            timeout=timeout,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    def _get_worker_init_fn(self, actual_worker_id: int):
        """Wrap worker_init_fn to change the input `worker_id` for each worker. This is necessary because we
        use `num_workers` independent DataLoaders instead of one DataLoader with `num_workers` workers."""
        if self.worker_init_fn is None:
            return None

        worker_init_fn = self.worker_init_fn

        def wrapped_worker_init_fn(dataloader_worker_id: int):
            worker_init_fn(dataloader_worker_id + actual_worker_id)

        return wrapped_worker_init_fn

    def get_stream_loaders(self) -> Generator[List[AudioSample], None, None]:
        if isinstance(self.dataset, IterableDataset):
            num_workers = max(1, self.num_workers)
            datasets = self.dataset.split(
                max_workers=num_workers, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last
            )
        else:
            datasets = self.dataset

        # TODO (JDH): Pass `collate_fn` to the DataLoaders to collate each worker's partial batch before returning it
        #             to the main process to be collated into a full batch. Is this faster than collating all in the
        #             main process?
        stream_loaders = [
            DataLoader(dataset, **self.dataloader_kwargs, worker_init_fn=self._get_worker_init_fn(i))
            for i, dataset in enumerate(datasets)
        ]

        if self.drop_last:
            # zip only where all streams have data and set is_last on all AudioSamples in last batch.
            stream_loaders = zip(*stream_loaders)
            stream_loaders = force_is_last_on_last_batch(stream_loaders)
        else:
            # zip longest and filter out None values which are returned when a stream has run out of data.
            stream_loaders = itertools.zip_longest(*stream_loaders)
            stream_loaders = (filter(lambda x: x is not None, batch_parts) for batch_parts in stream_loaders)

        # flatten batches from Generator[List[List[AudioSample]]] to Generator[List[AudioSample]].
        stream_loader = (list(itertools.chain.from_iterable(batch_parts)) for batch_parts in stream_loaders)

        return stream_loader

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            collated_batch = self.collate_fn(batch_parts)
            yield collated_batch


if __name__ == "__main__":
    source_file = pathlib.Path("/m2/research/source/wsj/test_dev93.txt")
    source_df = pd.read_csv(source_file)
    source_df.filename = source_df.filename.apply(lambda x: x + ".wav")

    transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80)
    # transform = None
    datasets = AudioStreamDataset(file_list=source_df.filename, chunk_seconds=1.0, transform=transform)
    loader = MultiStreamDataLoader(
        datasets,
        batch_size=8,
        num_workers=4,
        shuffle=True,
        drop_last=False,
        collate_fn=AudioStreamDataset.custom_collate,
    )

    for epoch in range(1):
        batches = []
        files_seen = set()
        samples_seen = []
        ts = time.time()
        for batch in loader:
            batches.append(batch)

            files_seen.update(batch.stream_state.ids)
            sample_ids = [
                f"{batch.stream_state.ids[i]} {batch.stream_state.chunk_index[i]:2d}"
                for i in range(len(batch.stream_state.ids))
            ]
            samples_seen.extend(sample_ids)

            filenames = [id.split("/")[-1].split(".")[0] for id in batch.stream_state.ids]
            chunk_idx = [chunk_index for chunk_index in batch.stream_state.chunk_index]
            is_first = [int(is_first) for is_first in batch.stream_state.is_first]
            is_last = [int(is_last) for is_last in batch.stream_state.is_last]

            filenames_with_index = [f"{filename} {idx:2d}" for filename, idx in zip(filenames, chunk_idx)]
            print(filenames_with_index, is_first, is_last)

        print(f"Time taken: {time.time() - ts:.2f} s")

        print(f"Expected files: {len(source_df)}")
        print(f"Files seen: {len(files_seen)}")
        print(
            f"Expected samples: ",
            sum([math.ceil(ln / datasets.chunk_seconds) for ln in datasets.file_lengths.values()]),
        )
        print(f"Samples seen: {len(samples_seen)}")

    import IPython
    IPython.embed(using=False)
