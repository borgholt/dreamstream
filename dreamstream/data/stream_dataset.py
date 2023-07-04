import collections
import itertools
import logging

from typing import Any, Dict, List, Callable, Optional, Generator, Tuple, Union
from tqdm import tqdm

import random
import numpy as np

import torch
import torchaudio

from torch.utils.data import IterableDataset, DataLoader

from dreamstream.data.data_objects import StreamSample
from dreamstream.tensor import StreamTensor, as_stream_tensor, stream_metadata


LOGGER = logging.getLogger(__file__)


# TODO (JDH): Might be worthwile implementing this with numpy and compiling with numba.
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


def concatenate_ragged(
    batch: List[torch.Tensor], pad_value: float = 0, dim: int = -1
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Concatenate a batch of tensors of some shape (*, dim, *) where `dim` may have different size.

    The output tensor has shape (batch, *, dim, *)."""
    # convert to positive dim.
    dim = dim if dim >= 0 else len(batch[0].shape) + dim

    # move temporal dimension to beginning and pad with torch.nn.utils.rnn.pad_sequence.
    batch = [b.transpose(dim, 0) for b in batch]  # (dim, *)
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_value)  # (batch, dim, *)
    batch = batch.transpose(dim + 1, 1)  # (batch, *, dim, *)
    return batch


class StreamDataset(IterableDataset):
    """Interface for datasets that can be used with the `MultiStreamDataLoader`."""

    def custom_collate(self, batch: List[StreamSample], names: List[str]) -> StreamTensor:
        """An optional method used to collate a list of AudioSamples into a StreamTensor.

        NOTE: If a Callable is passed as the `custom_collate` argument to the `MultiStreamDataLoader`, this method
        does not need to be implemented.

        NOTE: If this method is not implemented, and `custom_collate` is not passed to the `MultiStreamDataLoader`,
        the default collate function will be used.
        """
        raise NotImplementedError

    def split(self, num_streams: int, shuffle: bool = False) -> List["StreamDataset"]:
        """An optional method used to split the dataset into `num_streams` datasets for a MultiStreamDataLoader.

        NOTE: If a list of datasets is already available, this can be passed to the `MultiStreamDataLoader` instead and
        this method does not need to be implemented.

        Args:
            num_streams (int): The number of datasets to split into.
            shuffle (bool, optional): If True, shuffle the dataset before splitting. Defaults to False.

        Returns:
            List[StreamDataset]: A list of `num_streams` datasets.
        """
        raise NotImplementedError


class AudioStreamDataset(StreamDataset):
    def __init__(
        self,
        file_list: List[str],
        chunk_seconds: float = 1.0,
        file_lengths: List[Union[float, int]] = None,
        transform: Callable = None,
        names: List[str] = None,
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

        Each files is read only once per iteration.

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
        self.names = names
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if file_lengths is None:
            self.file_metadata = file_metadata or self.get_file_metadata()  # TODO: Delete?
            self.file_lengths = {file: m.num_frames / m.sample_rate for file, m in self.file_metadata.items()}
        else:
            self.file_metadata = dict()
            self.file_lengths = {file: length for file, length in zip(file_list, file_lengths)}

    @property
    def batch_size(self) -> int:
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

    @staticmethod
    def force_is_last_on_last_batch(
        stream: Generator[List[StreamSample], None, None]
    ) -> Generator[List[StreamSample], None, None]:
        # yield all but last batch
        to_yield = next(stream)
        for value in stream:
            yield to_yield
            to_yield = value

        # yield last batch with eos=True on all samples
        for sample in to_yield:
            sample.eos = True

        yield to_yield

    def process_data(self, file: str) -> Generator[StreamSample, None, None]:
        # TODO (JDH): For vseq, we can generalize this to support several modalities with Loader classes by adding
        #   a Chunker class per modality that splits a loaded file into chunks.
        # Load file
        audio, sample_rate = torchaudio.load(file)  # (C, L)
        num_frames = audio.shape[-1]

        # Apply transforms
        if self.transform:
            audio = self.transform(audio)

        # Chunk transformed audio
        stride = num_frames / audio.shape[-1]
        chunk_size = round(self.chunk_seconds * sample_rate / stride)
        audio_chunks = [audio[..., i : i + chunk_size] for i in range(0, audio.shape[-1], chunk_size)]

        for i, audio_chunk in enumerate(audio_chunks):
            audio_sample = StreamSample(
                data=audio,
                sos=(i == 0),
                eos=(i == len(audio_chunks) - 1),
                length=audio_chunk.shape[-1],
                chunk_index=i,
                num_chunks=len(audio_chunks),
                file=file,
                file_metadata=self.file_metadata.get(file, None),
                id=str(file),
            )
            yield audio_sample

    def get_stream(self, file_list: List[str]) -> Generator[StreamSample, None, None]:
        """Create a stream of data from a number of files by processing each file with `self.process_data`."""
        return itertools.chain.from_iterable(map(self.process_data, file_list))

    def get_streams(self) -> Generator[List[StreamSample], None, None]:
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

        # Partition file list into batch_size file lists (from start plus offset to end in steps of batch_size).
        data_streams = [self.get_stream(file_list[i :: self.batch_size]) for i in range(self.batch_size)]
        if self.drop_last:
            data_stream = zip(*data_streams)
            data_stream = self.force_is_last_on_last_batch(data_stream)
        else:
            data_stream = itertools.zip_longest(*data_streams)
            data_stream = (list(filter(lambda x: x is not None, batch_parts)) for batch_parts in data_stream)

        return data_stream

    def __iter__(self) -> Generator[List[StreamSample], None, None]:
        return self.get_streams()

    def custom_collate(self, batch: List[StreamSample]) -> StreamTensor:
        """Collate a batch of AudioSample instances into a StreamTensor which is a torch.Tensor wth a StreamMetadata."""
        # Sort by length.
        batch = sorted(batch, key=lambda x: x.length, reverse=True)

        data = [sample.data for sample in batch]
        data = concatenate_ragged(data, dim=-1)

        # Collate batch metadata
        ids = [sample.id for sample in batch]
        sos = torch.tensor([sample.sos for sample in batch])
        eos = torch.tensor([sample.eos for sample in batch])
        lengths = torch.tensor([sample.length for sample in batch])
        chunk_indices = torch.tensor([sample.chunk_index for sample in batch])

        # Create stream tensor and its meta.
        meta = stream_metadata(ids, sos, eos, lengths, chunk_indices)
        batch = as_stream_tensor(data, meta, self.names)
        return batch

    def split(self, num_workers: int, batch_size: int, shuffle: bool, drop_last: bool) -> List["AudioStreamDataset"]:
        """Split the dataset into multiple datasets, each with a unique subset of the data."""
        # Validate number of workers and change it not a divisor of batch size.
        if num_workers <= 0:
            raise ValueError(f"num_workers must be greater than 0 but was {num_workers}")

        if batch_size % num_workers != 0:
            raise ValueError(f"`num_workers` must be a divisor of batch_size ({batch_size}) but got {num_workers}.")

        # Split file_list among num_workers such that each worker gets approximately the same total number of chunks.
        file_lengths = [self.file_lengths[file] for file in self.file_list]
        num_chunks_per_file = [round(length / self.chunk_seconds) for length in file_lengths]
        partitioned_num_chunks, indices = partition_by_sum(num_chunks_per_file, num_workers)
        file_lists = [[self.file_list[i] for i in ids] for ids in indices]

        if self.file_metadata:
            file_metadatas = [{file: self.file_metadata[file] for file in partition} for partition in file_lists]
        else:
            file_metadatas = [dict() for _ in file_lists]

        batch_size_per_worker = batch_size // num_workers

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
            for i in range(num_workers)
        ]
        return datasets


def continue_buffering(
    is_stream_empty: bool,
    active_ids: List[str],
    buffers: Dict[str, collections.deque],
    batch_size: int,
) -> bool:
    if is_stream_empty:
        return False

    num_current_running = len(active_ids)
    num_current_buffered = sum(1 for id in active_ids if id in buffers and buffers[id] and buffers[id][-1].eos)
    num_next_buffered = sum(1 for id in buffers.keys() if id not in active_ids and buffers[id] and buffers[id][-1].eos)

    return num_current_buffered < num_current_running or num_next_buffered < (batch_size - num_current_running)


def make_streams_synchronous(
    stream: Generator[List[StreamSample], None, None]
) -> Generator[List[StreamSample], None, None]:
    """
    Raises:
        StopIteration: When the stream is exhausted.
    """
    stream = iter(stream)
    batch = next(stream)

    batch_size = len(batch)  # Assumes that first batch has the full batch size
    active_ids = [sample.id for sample in batch]

    buffers = collections.defaultdict(collections.deque)
    buffers.update({sample.id: collections.deque([sample]) for sample in batch})

    is_stream_empty = False
    while True:
        # Make sure the active files are buffered up until and including their eos chunks.
        while continue_buffering(is_stream_empty, active_ids, buffers, batch_size):
            # Buffer up until at least `batch_size` files different from the active ones have been buffered until eos.
            try:
                for sample in next(stream):
                    buffers[sample.id].append(sample)
            except StopIteration:
                is_stream_empty = True

        # Create batch and yield if not empty.
        batch = [buffers[id].popleft() for id in active_ids if buffers[id]]
        if batch:
            active_ids = [sample.id for sample in batch if not sample.eos]  # Update active ids to drop ended files.
            yield batch
        else:
            # All active ids have been removed and the batch was empty, so all active files have ended.
            # Try to get new ids from the buffer.
            if active_ids:
                raise RuntimeError("All active ids have been removed but the batch was empty.")

            active_ids = [k for k in buffers if buffers[k] and buffers[k][0].sos][:batch_size]

            # Remove empty buffers.
            filtered_buffers = {id: deque for id, deque in buffers.items() if deque}
            buffers.clear()
            buffers.update(filtered_buffers)

            if is_stream_empty and not active_ids:
                return None


class MultiStreamDataLoader(IterableDataset):
    def __init__(
        self,
        dataset: Union[IterableDataset, List[IterableDataset]],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        overlapping_batches: bool = False,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context: Any | None = None,
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
            batch_size (int): The batch size.
            shuffle (bool): Whether to shuffle the dataset between epochs.
            num_workers (int, optional): When a single dataset is given, it is split into `num_workers` subsets and
                each to be processed in parallel by a DataLoader using a single worker. When a list of datasets is
                given, `num_workers` has no effect. Defaults to 0.
            overlapping_batches (bool, optional): If True, each new batch will introduce new files to replace those that
                ended in the previous batch, keeping the batch size constant (except for the last batch if
                `drop_last=False`). If False, new files will only start once all files from the previous batch have
                ended. This is usually more memory and compute efficient for state management in models in online mode,
                but also leads to more batches. Defaults to False.
            drop_last (bool, optional): Drop the last batch(es) if smaller than `batch_size`. Defaults to False.
            collate_fn (Callable, optional): A function that takes a list of batch parts and collates them into a
                batch. Defaults to None.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.overlapping_batches = overlapping_batches
        self.drop_last = drop_last
        self.worker_init_fn = worker_init_fn
        self.collate_fn = collate_fn

        if drop_last and not overlapping_batches:
            raise ValueError("`drop_last` can only be used when overlapping batches.")

        if isinstance(dataset, list):
            if num_workers > 0:
                raise ValueError("When a list of datasets is given, `num_workers` has no effect.")
            self.num_workers = len(dataset)

        if collate_fn is None:
            if hasattr(dataset, "collate_fn"):
                collate_fn = dataset.collate_fn
            else:
                collate_fn = torch.utils.data._utils.collate.default_collate

        self.dataloader_kwargs = dict(
            batch_size=None,
            collate_fn=None,
            num_workers=(0 if self.num_workers == 0 else 1),
            pin_memory=pin_memory,
            timeout=timeout,
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    def _get_worker_init_fn(self, actual_worker_id: int):
        """Wrap worker_init_fn to change the input `worker_id` for each worker. This is necessary because we
        use `num_workers` independent DataLoaders with one worker each instead of one with `num_workers` workers."""
        if self.worker_init_fn is None:
            return None

        worker_init_fn = self.worker_init_fn

        def wrapped_worker_init_fn(dataloader_worker_id: int):
            worker_init_fn(dataloader_worker_id + actual_worker_id)

        return wrapped_worker_init_fn

    def get_stream_loaders(self) -> Generator[Any, None, None]:
        # TODO (JDH): Wrap this method in a dedicated process to offload all data preparation from main process.
        # Currently this happens in the main proces and may take a bit of time, especially the collation.

        if isinstance(self.dataset, IterableDataset):
            num_workers = max(1, self.num_workers)
            datasets = self.dataset.split(
                num_workers=num_workers, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last
            )
        else:
            datasets = self.dataset

        # TODO (JDH): Pass `collate_fn` to the DataLoaders to collate each worker's partial batch before returning it
        #   to the main process to be collated into a full batch. Is this faster than collating all in the main process?
        stream_loaders = [
            DataLoader(dataset, **self.dataloader_kwargs, worker_init_fn=self._get_worker_init_fn(i))
            for i, dataset in enumerate(datasets)
        ]  # List[DataLoader] == List[Generator[List[StreamSample]]]

        if self.drop_last:
            # Zip only batch parts where all streams have data.
            # List[Generator[List[StreamSample]]] -> Generator[List[List[StreamSample]]]]
            stream_loaders = zip(*stream_loaders)
        else:
            # Zip all batch parts and filter out None values which are returned when a stream has run out of data.
            # List[Generator[List[StreamSample]]] -> Generator[List[List[StreamSample]]]]
            stream_loaders = itertools.zip_longest(*stream_loaders, fillvalue=None)
            stream_loaders = (filter(lambda x: x is not None, batch_parts) for batch_parts in stream_loaders)

        # Flatten batches from loaders and datasets to single generator i.e. from Generator[List[List[StreamSample]]] to
        # Generator[List[StreamSample]].
        stream_loader = (list(itertools.chain.from_iterable(batch_parts)) for batch_parts in stream_loaders)

        # Wait for all files being streamed to finish before starting the next set of files.
        if not self.overlapping_batches:
            stream_loader = make_streams_synchronous(stream_loader)

        # Collate batches
        stream_loader = (self.collate_fn(batch) for batch in stream_loader)
        return stream_loader

    def __iter__(self) -> Generator[Any, None, None]:
        for batch in self.get_stream_loaders():
            yield batch


class MultiStreamOneProcessDataLoader:
    def __init__(
        self,
        dataset: Union[IterableDataset, List[IterableDataset]],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        overlapping_batches: bool = False,
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

        It works by wrapping _MultiStreamDataLoader in a DataLoader to perform the collation of batches in a single
        process separate to the main process.

        Args:
            dataset (Union[IterableDataset, List[IterableDataset]]): A single or a list of IterableDataset instances.
            batch_size (int): The batch size.
            num_workers (int, optional): When a single dataset is given, it is split into `num_workers` subsets and
                each to be processed in parallel by a single worker. When a list of datasets is given, `num_workers`
                has no effect. Defaults to 0.
            overlapping_batches (bool, optional): If True, the batches are non-overlapping. In this case, new files
                are started only once every file in the batch has ended. If False, a new file is started as soon as a
                file in the previous batch ended. Defaults to False.
            drop_last (bool, optional): Drop the last batch(es) if smaller than `batch_size`. Defaults to False.
            collate_fn (Callable, optional): A function that takes a list of batch parts and collates them into a
                batch. Defaults to None.
        """
        if num_workers > 1:
            raise ValueError("num_workers > 1 is not supported.")

        self.iterable_dataloader = MultiStreamDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            overlapping_batches=overlapping_batches,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            prefetch_factor=None,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

        self.dataloader = DataLoader(
            self.iterable_dataloader,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=None,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    def __iter__(self) -> Generator[Any, None, None]:
        for batch in self.dataloader:
            yield batch
