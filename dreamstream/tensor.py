import itertools
import warnings

from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple, Union, Mapping, Sequence

import torch
import numpy as np

from torch import Tensor

from dreamstream.func_coverage import DECOUPLE_FUNCTIONS, RECOUPLE_FUNCTIONS, VALID_FUNCTIONS, OVERRIDDEN_FUNCTIONS
from dreamstream.utils.flags import BATCH, LENGTH
from dreamstream.utils.numba import (
    is_sorted_ascending,
    make_indices_positive_,
    minmax,
    update_eos_from_integer,
    update_eos_from_slice,
    update_lengths_from_list_of_indices,
)


# TODO (JDH): Make StreamMetadata methods like cat, split and index lazily evaluated such that they only evaluate when
# they are needed. This minimizes overhead computation on StreamTensors that end up as leaf nodes in the graph.


def decouple_recursive(x, metas: Optional[List["StreamMetadata"]] = None, names: Optional[List[str]] = None):
    """Recurse a nested structure and decouple all StreamTensors."""
    if isinstance(x, StreamTensor):
        if metas is None and names is None:
            return x.tensor()

        tensor, meta, name = x.decouple()
        metas.append(meta)
        names.append(name)
        return tensor

    elif isinstance(x, Mapping):
        return type(x)((k, decouple_recursive(v, metas=metas, names=names)) for k, v in x.items())
    elif isinstance(x, Sequence):
        return type(x)(decouple_recursive(v, metas=metas, names=names) for v in x)

    return x


def decouple(func, *args, **kwargs):
    """Call function on tensor after decoupling it from StreamMetadata."""
    args, kwargs = decouple_recursive((args, kwargs))
    return func(*args, **kwargs)


def recouple(func, *args, **kwargs):
    """Call function on tensor after recoupling it to StreamMetadata and recouple again afterwards."""
    metas, names = [], []
    args, kwargs = decouple_recursive((args, kwargs), metas=metas, names=names)
    if not all(metas[0] == m for m in metas):
        raise ValueError("All StreamTensors must have the same StreamMetadata.")

    tensor = func(*args, **kwargs)
    return as_stream_tensor(data=tensor, meta=metas[0], names=names[0])


class LazyProxy(object):
    """A proxy class that lazily instantiates an object of type cls with arguments *args and **kwargs."""

    def __init__(self, cls, *args, **kwargs):
        self.__dict__["_cls"] = cls
        self.__dict__["_args"] = args
        self.__dict__["_kwargs"] = kwargs
        self.__dict__["_obj"] = None

    def __getattr__(self, name):
        if self.__dict__["_obj"] is None:
            self.__init_obj()

        return getattr(self.__dict__["_obj"], name)

    def __setattr__(self, name, value):
        if self.__dict__["_obj"] is None:
            self.__init_obj()

        setattr(self.__dict__["_obj"], name, value)

    def __getitem__(self, key):
        if self.__dict__["_obj"] is None:
            self.__init_obj()

        return self.__dict__["_obj"].__getitem__(key)

    def __copy__(self):
        if self.__dict__["_obj"] is None:
            self.__init_obj()

        return self.__dict__["_obj"].__copy__()

    def __eq__(self, other):
        if self.__dict__["_obj"] is None:
            self.__init_obj()

        return self.__dict__["_obj"].__eq__(other)

    def __len__(self):
        if self.__dict__["_obj"] is None:
            self.__init_obj()

        return self.__dict__["_obj"].__len__()

    def __repr__(self):
        if self.__dict__["_obj"] is None:
            return f"LazyProxy({self.__dict__['_cls'].__name__}, {self.__dict__['_args']}, {self.__dict__['_kwargs']})"
        return self.__dict__["_obj"].__repr__()

    def __init_obj(self):
        self.__dict__["_obj"] = object.__new__(self.__dict__["_cls"])
        self.__dict__["_obj"].__init__(*self.__dict__["_args"], **self.__dict__["_kwargs"])


class LazyInit(object):
    """A class that lazily initializes its attributes."""

    def __new__(cls, *args, **kwargs):
        return LazyProxy(cls, *args, **kwargs)


class StreamMetadata(LazyInit):
    """Metadata associated with a batch of streamed input tensors."""

    __slots__ = [
        "ids",
        "_sos",
        "_eos",
        "_lengths",
        "_chunk_indices",
        "_min_length",
        "_max_length",
        "_lengths_updated",
        "_any_starting",
        "_any_ending",
        "_all_starting",
        "_all_ending",
        "_any_starting_or_ending",
        "_all_starting_and_ending",
        "_sos_or_eos_updated",
    ]

    def __init__(
        self,
        ids: Union[str, List[str]],
        sos: Union[bool, List[bool], torch.BoolTensor],
        eos: Union[bool, List[bool], torch.BoolTensor],
        lengths: Union[int, List[int], torch.IntTensor],
        chunk_indices: Optional[Union[int, List[int], torch.IntTensor]] = None,
        _copy_on_init: bool = False,
    ):
        super().__init__()

        if isinstance(ids, str):
            ids = [ids]
        if isinstance(lengths, int):
            lengths = [lengths]
        if isinstance(sos, bool):
            sos = [sos]
        if isinstance(eos, bool):
            eos = [eos]

        if not len(ids) == len(lengths) == len(sos) == len(eos):
            raise ValueError("ids, lengths, sos and eos must have the same length.")

        sos_tensor = torch.as_tensor(sos, dtype=torch.bool)
        eos_tensor = torch.as_tensor(eos, dtype=torch.bool)
        lengths_tensor = torch.as_tensor(lengths, dtype=torch.int)

        if _copy_on_init:
            if sos_tensor is sos:
                sos_tensor = sos_tensor.clone()
            if eos_tensor is eos:
                eos_tensor = eos_tensor.clone()
            if lengths_tensor is lengths:
                lengths_tensor = lengths_tensor.clone()

        if not all(isinstance(i, str) for i in ids):
            raise ValueError("ids must be a list of strings.")

        if lengths_tensor.ndim > 1 or eos_tensor.ndim > 1 or lengths_tensor.ndim > 1:
            raise ValueError("sos, eos and lengths must be 1-dimensional.")

        self.ids = ids
        self._sos = sos_tensor
        self._eos = eos_tensor
        self._lengths = lengths_tensor
        self._chunk_indices = chunk_indices

        self._min_length = None
        self._max_length = None
        self._lengths_updated = True

        self._any_starting = None
        self._any_ending = None
        self._all_starting = None
        self._all_ending = None
        self._any_starting_or_ending = None
        self._all_starting_and_ending = None
        self._sos_or_eos_updated = True

    @property
    def sos(self) -> torch.BoolTensor:
        return self._sos

    @sos.setter
    def sos(self, value: torch.BoolTensor):
        self._sos = value
        self._sos_or_eos_updated = True

    @property
    def eos(self) -> torch.BoolTensor:
        return self._eos

    @eos.setter
    def eos(self, value: torch.BoolTensor):
        self._eos = value
        self._sos_or_eos_updated = True

    @property
    def lengths(self) -> torch.IntTensor:
        return self._lengths

    @lengths.setter
    def lengths(self, lengths: torch.IntTensor):
        self._lengths = lengths
        self._lengths_updated = True

    @property
    def chunk_indices(self) -> Optional[torch.IntTensor]:
        return self._chunk_indices

    @chunk_indices.setter
    def chunk_indices(self, chunk_indices: Optional[torch.IntTensor]):
        self._chunk_indices = chunk_indices

    @property
    def min_length(self):
        """Return the minimum length of the batch and recompute only if lengths have been mutated."""
        self._update_lengths()
        return self._min_length

    @min_length.setter
    def min_length(self, length):
        raise AttributeError("min_length is read-only.")

    @property
    def max_length(self):
        """Return the maximum length of the batch and recompute only if lengths have been mutated."""
        self._update_lengths()
        return self._max_length

    @max_length.setter
    def max_length(self, length):
        raise AttributeError("max_length is read-only.")

    def _update_lengths(self):
        if self._lengths_updated:
            self._min_length = self.lengths.min().item()
            self._max_length = self.lengths.max().item()
            self._lengths_updated = False

    @property
    def starting_lengths(self) -> torch.IntTensor:
        return self.lengths[self.sos]

    @starting_lengths.setter
    def starting_lengths(self, lengths: Union[int, torch.IntTensor]):
        self._lengths[self.sos] = lengths
        self._lengths_updated = True

    @property
    def ending_lengths(self) -> torch.IntTensor:
        return self.lengths[self.eos]

    @ending_lengths.setter
    def ending_lengths(self, lengths: Union[int, torch.IntTensor]):
        self._lengths[self.eos] = lengths
        self._lengths_updated = True

    @property
    def any_starting(self) -> bool:
        self._update_logicals()
        return self._any_starting

    @property
    def any_ending(self) -> bool:
        self._update_logicals()
        return self._any_ending

    @property
    def all_starting(self) -> bool:
        self._update_logicals()
        return self._all_starting

    @property
    def all_ending(self) -> bool:
        self._update_logicals()
        return self._all_ending

    @property
    def any_starting_or_ending(self) -> bool:
        self._update_logicals()
        return self._any_starting_or_ending

    @property
    def all_starting_and_ending(self) -> bool:
        self._update_logicals()
        return self._all_starting_and_ending

    def _update_logicals(self):
        if self._sos_or_eos_updated:
            self._any_starting = self.sos.any().item()
            self._any_ending = self.eos.any().item()
            self._all_starting = self.sos.all().item()
            self._all_ending = self.eos.all().item()
            self._any_starting_or_ending = self._any_starting or self._any_ending
            self._all_starting_and_ending = self._all_starting and self._all_ending
            self._sos_or_eos_updated = False

    def __copy__(self):
        """Returns a deep of the StreamMetadata object because we don't support shallow copies."""
        new_meta = StreamMetadata(
            ids=self.ids,
            sos=self.sos,
            eos=self.eos,
            lengths=self.lengths,
        )
        new_meta.__dict__.update(self.__dict__)
        return new_meta

    def __deepcopy__(self):
        """Return a deep copy of the StreamMetadata object."""
        return StreamMetadata(
            ids=self.ids,
            sos=self.sos,
            eos=self.eos,
            lengths=self.lengths,
            chunk_indices=self._chunk_indices,
            _copy_on_init=True,
        )

    def __eq__(self, other: "StreamMetadata") -> bool:
        """Return True if the two StreamMetadata objects have the same ids, sos, eos and lengths, False otherwise."""
        return (
            self.ids == other.ids
            and self.sos.equal(other.sos)
            and self.eos.equal(other.eos)
            and self.lengths.equal(other.lengths)
            and (self._chunk_indices is None and other._chunk_indices is None) or self._chunk_indices.equal(other.lengths)
        )

    def __len__(self) -> int:
        """Return the number of elements in the batch."""
        return len(self.ids)

    def __repr__(self) -> str:
        """Returns a string representation of the StreamMetadata object shortening the ids, sos, eos and lengths
        if they are longer than 80 characters.

        StreamTensor(
            ids=["a", "b", ..., "c"],
            sos=[True, False, ..., False],
            eos=[False, False, ..., True],
            lengths=[10, 20, ..., 30],
        )
        """
        if sum(len(i) for i in self.ids) > 80:
            # Shorten the ids keeping some first and the last element.
            last = repr(self.ids[-1])
            repr_ids = [repr(self.ids[0])]
            length = len(repr_ids[0]) + len(last) + 7 + len(repr(self.ids[1]))
            i = 1
            while length < 80:
                id_repr = repr(self.ids[i])
                repr_ids.append(id_repr)
                length += len(repr(self.ids[i + 1])) + 2
                i += 1

            short_ids_repr = ", ".join(repr_ids) + ", ..., " + repr(last)
        else:
            short_ids_repr = repr(self.ids)

        return (
            "StreamMetadata(\n"
            f"    ids={short_ids_repr},\n"
            f"    sos={repr(self.sos)},\n"
            f"    eos={repr(self.eos)},\n"
            f"    lengths={repr(self.lengths)},\n"
            ")"
        )

    def __getitem__(
        self, indices: Union[int, slice, List[Any], Tuple[Any, ...], torch.IntTensor, torch.BoolTensor]
    ) -> "StreamMetadata":
        """Index the metadata along the batch and/or length dimensions."""
        return self.index(indices)

    def index(
        self, indices: Union[None, int, slice, List[Any], Tuple[Any, ...], torch.IntTensor, torch.BoolTensor]
    ) -> "StreamMetadata":
        """Index the metadata along the batch and/or length dimensions."""
        match indices:
            case list() | tuple() if not all(isinstance(i, (int, bool)) for i in indices):
                match indices:
                    case (batch_indices, None):
                        return self.index_batch(batch_indices)
                    case (None, length_indices):
                        return self.index_length(length_indices)
                    case (batch_indices, length_indices):
                        return self.index_batch(batch_indices).index_length(length_indices)
                    case (None, None):
                        return self.__deepcopy__()
            case torch.BoolTensor() if indices.ndim > 1:
                return self.index_batch_and_length(indices)
            case torch.Tensor() | int() | slice() | list():
                return self.index_batch(indices)
            case tuple() if len(indices) == 2:
                return self.index_batch(indices[0]).index_length(indices[1])
            case None:
                return self.__deepcopy__()
            case _:
                raise TypeError(f"Unsupported index type: {type(indices)}")

    def index_batch(
        self, indices: Union[None, int, slice, List[int], Tuple[int, ...], torch.IntTensor, torch.BoolTensor]
    ) -> "StreamMetadata":
        """Return a StreamMetadata object with the specified batch indices."""
        # TODO (JDH): Implement using match statement instead of if/else.
        if indices is None:
            return self.__deepcopy__()

        if isinstance(indices, torch.Tensor) and indices.ndim > 1:
            raise IndexError(f"Expected batch indices to be a 1-dimensional tensor, but got {indices.ndim} dimensions.")

        if isinstance(indices, torch.BoolTensor):
            ids = [id for i, id in enumerate(self.ids) if indices[i]]
            sos = self.sos[indices]
            eos = self.eos[indices]
            lengths = self.lengths[indices]
            chunk_indices = self._chunk_indices[indices] if self._chunk_indices is not None else None
            return StreamMetadata(ids, sos, eos, lengths, chunk_indices)

        if isinstance(indices, int):
            ids = [self.ids[indices]]
            sos = self.sos[[indices]]
            eos = self.eos[[indices]]
            lengths = self.lengths[[indices]]
            chunk_indices = self._chunk_indices[[indices]] if self._chunk_indices is not None else None
            return StreamMetadata(ids, sos, eos, lengths, chunk_indices)

        if isinstance(indices, slice):
            ids = self.ids[indices]
        else:  # List[int], Tuple[int, ...], torch.IntTensor
            ids = [self.ids[i] for i in indices]

        sos = self.sos[indices]
        eos = self.eos[indices]
        lengths = self.lengths[indices]
        chunk_indices = self._chunk_indices[indices] if self._chunk_indices is not None else None
        return StreamMetadata(ids, sos, eos, lengths, chunk_indices)

    def index_length(
        self, indices: Union[None, int, slice, List[int], Tuple[int], torch.IntTensor, torch.BoolTensor]
    ) -> "StreamMetadata":
        """Return a StreamMetadata object with the specified length indices."""
        match indices:
            case None:
                return self.__deepcopy__()
            case int():
                return self._index_length_int(indices)
            case slice():
                return self._index_length_slice(indices)
            case list() | tuple():
                return self._index_length_list(indices)
            case torch.Tensor() if indices.ndim == 1:
                return self._index_length_1d_tensor(indices)
            case torch.Tensor() if indices.ndim > 1:
                raise IndexError("Indexing length with multidimensional torch tensors is not supported.")
            case _:
                raise IndexError(f"Indexing length with {indices} is not supported.")

    def index_batch_and_length(self, indices: torch.BoolTensor) -> "StreamMetadata":
        """Return a StreamMetadata object with the batch and length indexed with a single 2D BoolTensor."""
        if isinstance(indices, torch.Tensor) and indices.ndim > 2:
            raise ValueError(f"Expected indices to be a 2-dimensional tensor, but got {indices.ndim} dimensions.")

        cumsum = indices.cumsum(dim=1)
        keep_ids = cumsum[:, -1] > 0

        if not keep_ids.any():
            return StreamMetadata([], torch.tensor([]), torch.tensor([]), torch.tensor([]))

        broadcast_batch = indices.size(0) == 1 and len(self.ids) > 1
        broadcast_length = indices.size(1) == 1 and self.max_length > 1
        if broadcast_batch and broadcast_length:
            return self.__deepcopy__()

        if not keep_ids.all():
            ids = [id for i, id in enumerate(self.ids) if keep_ids[i]]
            sos = self.sos[keep_ids]
            eos = self.eos[keep_ids]
            lengths = self.lengths[keep_ids]
            chunk_indices = self._chunk_indices[keep_ids] if self._chunk_indices is not None else None
            indices = indices[keep_ids]
        else:  # includes `broadcast_batch == True`
            ids = deepcopy(self.ids)
            sos = self.sos
            eos = self.eos
            lengths = self.lengths
            chunk_indices = self._chunk_indices

        sos = sos & indices[:, 0]  # SOS only if the first index is included.
        if not broadcast_length:
            # EOS if any index at or beyond the last non-padding index is included.
            start = cumsum[keep_ids, 0]
            stop = indices.size(-1) - torch.fliplr(indices).to(torch.float32).argmax(-1)
            eos = eos & (start < lengths) & (lengths <= stop)
            # eos = eos & indices[range(indices.size(0)), lengths - 1]  # EOS only if the last non-padding is included.
            lengths = cumsum[keep_ids, lengths - 1]
            chunk_indices = chunk_indices[keep_ids] if chunk_indices is not None else None

        return StreamMetadata(ids, sos, eos, lengths, chunk_indices)

    def _index_length_int(self, index: int) -> "StreamMetadata":
        # Convert negative indices to positive
        if index < 0:
            index = self.max_length + index

        # Set lengths to 1 for all examples except those where the integer index is in padding (set to 0)
        lengths = (self.lengths - index).clamp(min=0, max=1)
        sos = self.sos.clone() if index == 0 else torch.zeros_like(self.sos)
        # TODO (JDH): numba compiled arithmetic is much faster but slowed down due to conversion to/from numpy
        #   Maybe we should store sos and eos as numpy arrays instead of torch tensors?
        eos = torch.from_numpy(update_eos_from_integer(self.eos.numpy(), self.lengths.numpy(), index))
        chunk_indices = self._chunk_indices.clone() if self._chunk_indices is not None else None
        return StreamMetadata(deepcopy(self.ids), sos, eos, lengths, chunk_indices)

    def _index_length_slice(self, slice: slice) -> "StreamMetadata":
        # Convert start and stop to positive indices
        start, stop, stride = slice.indices(self.max_length)

        if stride < 0:
            # TODO (JDH): Allow negative strides only if all examples have the same length equal to tensor's length.
            msg = "Negative strides along length not supported on StreamTensors. Instead, use `reverse_sequence`."
            raise NotImplementedError(msg)

        if stride == 1:
            # TODO (JDH): Implement `update_lengths_from_integer`
            lengths = (self.lengths - start).clip(min=0, max=stop - start)
        else:
            # Compute the right-most length index that is included in the slice from slice(start, stop, stride)
            stop = self.max_length - ((self.max_length - 1 - start) % stride)
            lengths = (self.lengths - start).clip(min=0, max=stop - start).div(stride).ceil().to(self.lengths.dtype)

        sos = self.sos.clone() if start == 0 and stop > 0 else torch.zeros_like(self.sos)
        eos = torch.from_numpy(update_eos_from_slice(self.eos.numpy(), self.lengths.numpy(), start, stop))
        chunk_indices = self._chunk_indices.clone() if self._chunk_indices is not None else None
        return StreamMetadata(deepcopy(self.ids), sos, eos, lengths, chunk_indices)

    def _index_length_list(self, indices: Union[List[int], Tuple[int]]) -> "StreamMetadata":
        # Convert to numpy arrays for faster manipulation and numba jit support.
        lengths_np = self.lengths.numpy()
        indices_np = np.array(indices)
        make_indices_positive_(indices_np, self.max_length)  # inplace

        # Raise error if the indices are not sorted
        if not is_sorted_ascending(indices_np):
            raise IndexError("Indices must be sorted when indexing length with lists or tuples.")

        # Update lengths, sos, and eos
        min_i, max_i = minmax(indices_np)
        lengths = update_lengths_from_list_of_indices(lengths_np, indices_np)
        sos = self.sos.clone() if min_i == 0 else torch.zeros_like(self.sos)
        eos = torch.from_numpy(update_eos_from_slice(self.eos.numpy(), lengths_np, min_i, max_i))
        # TODO (JDH): Keep EOS true if the indexing spans over the last non-padding element.
        chunk_indices = self._chunk_indices.clone() if self._chunk_indices is not None else None
        return StreamMetadata(deepcopy(self.ids), sos, eos, lengths, chunk_indices)

    def _index_length_1d_tensor(self, indices: torch.Tensor) -> "StreamMetadata":
        if indices.dtype == torch.bool:
            return self._index_length_1d_booltensor(indices)
        return self._index_length_1d_inttensor(indices)

    def _index_length_1d_booltensor(self, indices: torch.BoolTensor) -> "StreamMetadata":
        return self._index_length_list(indices.nonzero().squeeze(1).tolist())  # Adds ~10 µs

    def _index_length_1d_inttensor(self, indices: torch.IntTensor) -> "StreamMetadata":
        return self._index_length_list(indices.tolist())  # Adds ~1 µs

    @classmethod
    def cat(cls, metas: List["StreamMetadata"], dim: str) -> "StreamMetadata":
        """Concatenate a list of StreamMetadata objects along a given dimension."""
        if dim == LENGTH:
            return cls.cat_length(metas)
        elif dim == BATCH:
            return cls.cat_batch(metas)
        else:
            raise ValueError(f"Invalid dimension: {dim}")

    @classmethod
    def cat_batch(cls, metas: List["StreamMetadata"]) -> "StreamMetadata":
        """Concatenate a list of StreamMetadata objects along the batch dimension.

        Args:
            metas (List[StreamMetadata]): The StreamMetadata objects to concatenate.

        Returns:
            StreamMetadata: The concatenated StreamMetadata object.
        """

        if not all(isinstance(s, StreamMetadata) for s in metas):
            raise TypeError("All objects in list must be of type StreamMetadata.")

        if len(metas) == 1:
            return deepcopy(metas[0])

        ids = list(itertools.chain.from_iterable([s.ids for s in metas]))
        sos = torch.cat([s.sos for s in metas], dim=0)
        eos = torch.cat([s.eos for s in metas], dim=0)
        lengths = torch.cat([s.lengths for s in metas], dim=0)
        if all(s.chunk_indices is not None for s in metas):
            chunk_indices = torch.cat([s.chunk_indices for s in metas], dim=0)
        else:
            chunk_indices = None
        return cls(ids, sos, eos, lengths, chunk_indices)

    @classmethod
    def cat_length(cls, metas: List["StreamMetadata"]) -> "StreamMetadata":
        """Concatenate a list of StreamMetadata objects along the length dimension.

        Args:
            metas (List[StreamMetadata]): The StreamMetadata objects to concatenate.

        Returns:
            StreamMetadata: The concatenated StreamMetadata object.
        """
        if len(metas) == 1:
            return deepcopy(metas[0])

        j = len(metas) - 1
        for i, s in enumerate(metas):
            if i != 0 and s.ids != metas[0].ids:
                raise ValueError("Cannot concatenate StreamMetadata objects with different ids.")
            if i != 0 and s.any_starting:
                raise ValueError('StreamMetadata objects where any "sos" is True should be first when concatenated.')
            if i != j and s.any_ending:
                raise ValueError('StreamMetadata objects where any "eos" is True should be last when concatenated.')

        ids = deepcopy(metas[0].ids)
        sos = metas[0].sos.clone()
        eos = metas[-1].eos.clone()
        lengths = sum([s.lengths for s in metas])
        if all(s.chunk_indices is not None for s in metas):
            chunk_indices = metas[-1].chunk_indices.clone()  # TODO (JDH): This assumes the right-most chunk is the last
        return cls(ids, sos, eos, lengths, chunk_indices)

    def split(self, split_size_or_sections: Union[int, List[int]], dim: str) -> List["StreamMetadata"]:
        """Split a StreamMetadata object into a list of StreamMetadata objects along a given dimension."""
        if dim == LENGTH:
            return self.split_length(split_size_or_sections)
        elif dim == BATCH:
            return self.split_batch(split_size_or_sections)
        raise ValueError(f"Invalid dimension: {dim}")

    def split_batch(self, split_size_or_sections: Union[int, List[int]]) -> List["StreamMetadata"]:
        """Split a StreamMetadata object into a list of StreamMetadata objects along the batch dimension.

        Args:
            split_size_or_sections (Union[int, List[int]]): Size of a single chunk or list of sizes for each chunk.

        Returns:
            List[StreamMetadata]: The split StreamMetadata objects.
        """
        if isinstance(split_size_or_sections, list) and sum(split_size_or_sections) != len(self):
            raise ValueError("Sum of split sizes must equal the size of the StreamMetadata object.")

        if isinstance(split_size_or_sections, int):
            start = range(0, len(self), split_size_or_sections)
            split_ids = [self.ids[i : i + split_size_or_sections] for i in start]
        else:
            slices = np.cumsum([0] + split_size_or_sections)
            split_ids = [self.ids[i:j] for i, j in zip(slices[:-1], slices[1:])]

        split_sos = self.sos.split(split_size_or_sections)
        split_eos = self.eos.split(split_size_or_sections)
        split_lengths = self.lengths.split(split_size_or_sections)
        split_chunk_indices = (
            self.chunk_indices.split(split_size_or_sections) if self.chunk_indices is not None else None
        )
        args_iter = zip(split_ids, split_sos, split_eos, split_lengths, split_chunk_indices)

        return [stream_metadata(*args) for args in args_iter]

    def split_length(self, split_size_or_sections: Union[int, List[int]]) -> List["StreamMetadata"]:
        """Split a StreamMetadata object into a list of StreamMetadata objects along the length dimension.

        Args:
            split_size_or_sections (Union[int, List[int]]): Size of a single chunk or list of sizes for each chunk.

        Returns:
            List[StreamMetadata]: The split StreamMetadata objects.
        """
        max_length = self.lengths.max().item()
        if isinstance(split_size_or_sections, list) and sum(split_size_or_sections) < max_length:
            raise ValueError("Sum of split sizes must be equal to (or larger than) the maximum length.")

        if isinstance(split_size_or_sections, int):
            start = np.arange(0, max_length, split_size_or_sections)
            end = (start + split_size_or_sections).clip(max=max_length)
        else:
            end = np.cumsum(split_size_or_sections)
            start = np.concatenate(([0], end[:-1]))

        return [self.index_length(slice(i, j)) for i, j in zip(start, end)]

    def unbind_batch(self) -> List["StreamMetadata"]:
        """Split a StreamMetadata object into a list of StreamMetadata objects along the batch dimension.

        Returns:
            List[StreamMetadata]: The split StreamMetadata objects.
        """
        return self.split_batch(1)

    def drop_empty(self) -> "StreamMetadata":
        """Drop any tensors that are empty."""
        return self.index_batch(self.lengths > 0)

    def clone(self) -> "StreamMetadata":
        """Return a deep copy of the StreamMetadata object."""
        return self.__deepcopy__()

    def size(self) -> int:
        """Return the size of the StreamMetadata object. Same as len()."""
        return len(self)


# TODO (JDH): Implement a MultiLengthStreamMetadata class that can handle multiple length dimensions.
class MultiLengthStreamMetadata:
    def __init__(self) -> None:
        raise NotImplementedError()


def stream_metadata(
    ids: Union[str, List[str]],
    sos: Union[bool, List[bool], torch.BoolTensor],
    eos: Union[bool, List[bool], torch.BoolTensor],
    lengths: Union[int, List[int], torch.IntTensor],
    chunk_indices: Optional[Union[int, List[int], torch.IntTensor]] = None,
) -> StreamMetadata:
    """Create a StreamMetadata object from the given arguments.

    Args:
        ids (Union[str, List[str]]): The ids of the input tensors.
        sos (bool): Whether the input tensors are the first in a batch.
        eos (bool): Whether the input tensors are the last in a batch.
        lengths (Union[int, List[int]]): The lengths of the input tensors.
        chunk_indices (int): The index of the chunk in the batch.
        num_chunks (Optional[int], optional): The number of chunks in the batch. Defaults to None.

    Returns:
        StreamMetadata: The created StreamMetadata object.
    """
    return StreamMetadata(
        ids=ids,
        sos=sos,
        eos=eos,
        lengths=lengths,
        chunk_indices=chunk_indices,
    )


class StreamTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, meta: StreamMetadata, *args, **kwargs) -> "StreamTensor":
        """Return a new StreamTensor object."""
        return super().__new__(cls, data, *args, **kwargs)

    def __init__(self, data, meta: StreamMetadata, *args, names: List[str] = None, **kwargs):
        """Initialize a StreamTensor object (self is StreamTensor, data is e.g. torch.Tensor)."""
        super().__init__()
        self.meta = meta
        if names is not None:
            self.rename_(*names)

    def __getstate__(self) -> Tuple[torch.Tensor, StreamMetadata, List[str]]:
        """Return the state of the StreamTensor object."""
        return self.decouple()

    def __setstate__(self, state: Tuple[torch.Tensor, StreamMetadata, List[str]]):
        """Set the state of the StreamTensor object."""
        self.__init__(*state)

    @classmethod
    def __torch_function__(cls, func: Callable, types: List[torch.Tensor], args=(), kwargs=None):
        """Custom __torch_function__ implementation for StreamTensor.

        Args:
            func (Callable): The intercepted torch function.
            types (List[torch.Tensor]): Types of any Tensor-like arguments.
            args (tuple, optional): Arguments to the intercepted torch function. Defaults to ().
            kwargs (_type_, optional): Key-word arguments to the intercepted torch function. Defaults to None.

        Raises:
            RuntimeError: If the intercepted function could not be handled safely.
        """
        if kwargs is None:
            kwargs = dict()

        if func in OVERRIDDEN_FUNCTIONS:
            # print(f"\n\n{func.__name__}: STREAM_TENSOR_FUNCTIONS\n\n")
            return OVERRIDDEN_FUNCTIONS[func](*args, **kwargs)

        if func in RECOUPLE_FUNCTIONS:
            # print(f"\n\n{func.__name__}: RECOUPLE_FUNCTIONS\n\n")
            return recouple(func, *args, **kwargs)

        if func in DECOUPLE_FUNCTIONS:
            # print(f"\n\n{func.__name__}: DECOUPLE_FUNCTIONS\n\n")
            return decouple(func, *args, **kwargs)

        if func in VALID_FUNCTIONS:
            # print(f"\n\n{func.__name__}: VALID_FUNCTIONS\n\n")
            return super().__torch_function__(func, types, args, kwargs)

        # Unhandled functions are passed to the torch.Tensor.__torch_function__ method.
        warnings.warn(
            f"Function `{func.__name__}` is not handled by `StreamTensor.__torch_function__` "
            "and may not work as expected."
        )
        out = super().__torch_function__(func, types, args, kwargs)

        metas = [x.meta for x in [*args, *kwargs.values()] if isinstance(x, StreamTensor)]
        if not all(s == metas[0] for s in metas[1:]):
            msg = (
                f"Called a torch function ({func.__name__}) which was not handled by "
                f"StreamTensor.__torch_function__ with {len(metas)} StreamTensors in the input."
                f"In this case the function can only be handled if the StreamTensors have equal metadata,"
                f"but they were not equal."
            )
            raise RuntimeError(msg)

        if isinstance(out, torch.Tensor):
            return StreamTensor(out, meta=metas[0])

        return out

    @property
    def has_batch_dim(self) -> bool:
        return BATCH in self.names

    @property
    def has_length_dim(self) -> bool:
        return LENGTH in self.names

    @property
    def batch_dim(self) -> int:
        return self.names.index(BATCH)

    @property
    def length_dim(self) -> int:
        return self.names.index(LENGTH)

    def is_batch_dim(self, dim: int) -> bool:
        return self.names[dim] == BATCH

    def is_length_dim(self, dim: int) -> bool:
        # TODO: Refine to include multiple length dims
        return self.names[dim] == LENGTH

    def batch_size(self):
        return self.size(self.batch_dim)

    def max_length(self):
        # TODO: Refine to include multiple length dims
        return self.size(self.length_dim)

    def tensor(self, keep_names: bool = False) -> torch.Tensor:
        """Return the underlying torch.Tensor."""
        tensor = torch.Tensor(self)  # 1-2 µs
        if not keep_names:
            tensor.rename_(None)  # 2-3 µs
        return tensor

    def drop_empty(self) -> "StreamTensor":
        """Remove empty tensors from the batch."""
        if self.meta.min_length > 0:
            return self
        if self.meta.max_length == 0:
            return None
        if len(self.meta) == 1 and self.meta.max_length > 0:
            return self
        tensor, meta, names = self.decouple()
        batch_dim = names.index(BATCH)
        tensor = torch.index_select(tensor, batch_dim, meta.lengths.nonzero().squeeze())
        return as_stream_tensor(data=tensor, meta=meta.drop_empty(), names=names)

    def named_tensor(self) -> torch.Tensor:
        """Return the underlying torch.Tensor with names."""
        return self.tensor(keep_names=True)

    def unpad_sequence(self, keep_names: bool = False) -> List["StreamTensor"]:
        """Remove padding along the specified dimension."""
        batch_dim = self.names.index(BATCH)
        length_dim = self.names.index(LENGTH)
        if batch_dim < length_dim:
            length_dim -= 1
        return [x.narrow(length_dim, 0, x.meta.lengths.item()) for x in self.unbind(dim=batch_dim)]

    def decouple(self, copy_meta: bool = False) -> Tuple[Tensor, StreamMetadata, Tuple[str]]:
        """Decouple the StreamTensor from names and metadata."""
        meta = self.meta.clone() if copy_meta else self.meta
        return self.tensor(), meta, self.names


def as_stream_tensor(
    data,
    meta: StreamMetadata,
    names: Tuple[Union[None, int]] = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> StreamTensor:
    """Convert a tensor to a StreamTensor. See also `torch.as_tensor`."""
    data = torch.as_tensor(data, dtype=dtype, device=device)
    if names:
        data = data.refine_names(*names)  # Make the tensor named if it isn't already.
    return StreamTensor(data=data, meta=meta)


def stream_tensor(
    data,
    meta: StreamMetadata,
    names: Tuple[Union[None, int]],
    dtype: torch.dtype = None,
    device: torch.device = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> StreamTensor:
    """Convert a tensor to a StreamTensor. See also `torch.tensor`."""
    if isinstance(data, torch.Tensor) and data.names != names:
        data = data.rename(*names)

    data = torch.tensor(
        data, names=names, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory
    )
    return StreamTensor(data=data, meta=meta)
