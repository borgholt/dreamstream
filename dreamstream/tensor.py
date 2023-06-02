import itertools
import warnings

from copy import deepcopy
from typing import Any, Callable, List, Tuple, Union

import torch
import numpy as np

from torch import Tensor

from dreamstream.func_coverage import DECOUPLE_FUNCTIONS, RECOUPLE_FUNCTIONS, VALID_FUNCTIONS, OVERRIDDEN_FUNCTIONS
from dreamstream.utils.flags import BATCH, LENGTH
from dreamstream.utils.numba import is_sorted_ascending, make_indices_positive_, minmax, update_eos_from_integer, update_lengths_from_list_of_indices


# TODO (JDH): Make StreamMetadata methods like cat, split and index lazily evaluated such that they only evaluate when
# they are needed. This minimizes overhead computation on StreamTensors that end up as leaf nodes in the graph.


def decouple(func, tensor, *args, **kwargs):
    """Call function on tensor after decoupling it from StreamMetadata."""
    return func(tensor.tensor(), *args, **kwargs)


def recouple(func, tensor, *args, **kwargs):
    """Call function on tensor after recoupling it to StreamMetadata and recouple again afterwards."""
    tensor, meta, names = tensor.decouple()
    tensor = func(tensor, *args, **kwargs)
    tensor.rename_(*names)
    return as_stream_tensor(data=tensor, meta=meta, names=names)


class StreamMetadata:
    """Metadata associated with a batch of streamed input tensors."""

    def __init__(
        self,
        ids: Union[str, List[str]],
        sos: Union[bool, List[bool], torch.BoolTensor],
        eos: Union[bool, List[bool], torch.BoolTensor],
        lengths: Union[int, List[int], torch.IntTensor],
    ):
        # TODO: Make initialization lazy such that it only happens when the StreamMetadata is actually used.
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

        sos = torch.as_tensor(sos, dtype=torch.bool)
        eos = torch.as_tensor(eos, dtype=torch.bool)
        lengths = torch.as_tensor(lengths, dtype=torch.int)

        if not all(isinstance(i, str) for i in ids):
            raise ValueError("ids must be a list of strings.")

        if sos.ndim > 1 or eos.ndim > 1 or lengths.ndim > 1:
            raise ValueError("sos, eos and lengths must be 1-dimensional.")

        self.ids = ids
        self._sos = sos
        self._eos = eos
        self._lengths = lengths

        self._min_length = None
        self._max_length = None
        self._lengths_updated = True
        self._update_lengths()

        self._any_starting = None
        self._any_ending = None
        self._all_starting = None
        self._all_ending = None
        self._any_starting_or_ending = None
        self._all_starting_and_ending = None
        self._sos_or_eos_updated = True
        self._update_logicals()

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
        """Return a deep copy of the StreamMetadata object."""
        return StreamMetadata(
            ids=deepcopy(self.ids),
            sos=self.sos.clone(),
            eos=self.eos.clone(),
            lengths=self.lengths.clone(),
        )

    def __eq__(self, other: "StreamMetadata") -> bool:
        """Return True if the two StreamMetadata objects have the same ids, sos, eos and lengths, False otherwise."""
        return (
            self.ids == other.ids
            and self.sos.equal(other.sos)
            and self.eos.equal(other.eos)
            and self.lengths.equal(other.lengths)
        )

    def __len__(self) -> int:
        """Return the number of elements in the batch."""
        return len(self.ids)

    def __repr__(self) -> str:
        # TODO: Maybe make a more verbose __repr__. For example:
        # StreamTensor(
        #     ids=["a", "b", ..., "c"],
        #     sos=[True, False, ..., False],
        #     eos=[False, False, ..., True],
        #     lengths=[10, 20, ..., 30],
        # )
        return f"StreamMetadata(size={self.size()})"

    def __getitem__(
        self, indices: Union[int, slice, List[Any], Tuple[Any, ...], torch.IntTensor, torch.BoolTensor]
    ) -> "StreamMetadata":
        """Index the metadata along the batch and/or length dimensions."""
        return self.index(indices)

    def index(
        self, indices: Union[None, int, slice, List[Any], Tuple[Any, ...], torch.IntTensor, torch.BoolTensor]
    ) -> "StreamMetadata":
        """Index the metadata along the batch and/or length dimensions."""
        if isinstance(indices, (list, tuple)) and not all(isinstance(i, (int, bool)) for i in indices):
            batch_indices, length_indices = indices
            if batch_indices is None and length_indices is None:
                return self
            if batch_indices is not None and length_indices is None:
                return self.index_batch(batch_indices)
            if batch_indices is None and length_indices is not None:
                return self.index_length(length_indices)
            return self.index_batch(batch_indices).index_length(length_indices)

        return self.index_batch(indices)            

    def index_batch(
        self, indices: Union[None, int, slice, List[int], Tuple[int, ...], torch.IntTensor, torch.BoolTensor]
    ) -> "StreamMetadata":
        """Return a StreamMetadata object with the specified batch indices."""
        if indices is None:
            return self  # TODO (JDH): Should we return a copy instead?

        if isinstance(indices, torch.Tensor) and indices.ndim > 1:
            raise ValueError(f"Expected batch indices to be a 1-dimensional tensor, but got {indices.ndim} dimensions.")

        if isinstance(indices, torch.BoolTensor):
            ids = [id for i, id in enumerate(self.ids) if indices[i]]
            sos = self.sos[indices]
            eos = self.eos[indices]
            lengths = self.lengths[indices]
            return StreamMetadata(ids, sos, eos, lengths)

        if isinstance(indices, int):
            ids = [self.ids[indices]]
            sos = self.sos[[indices]]
            eos = self.eos[[indices]]
            lengths = self.lengths[[indices]]
            return StreamMetadata(ids, sos, eos, lengths)

        if isinstance(indices, slice):
            ids = self.ids[indices]
        else:  # List[int], Tuple[int, ...], torch.IntTensor
            ids = [self.ids[i] for i in indices]

        sos = self.sos[indices]
        eos = self.eos[indices]
        lengths = self.lengths[indices]
        return StreamMetadata(ids, sos, eos, lengths)

    def index_length(
        self, indices: Union[None, int, slice, List[int], Tuple[int], torch.IntTensor, torch.BoolTensor]
    ) -> "StreamMetadata":
        """Return a StreamMetadata object with the specified length indices."""
        match indices:
            case None:
                return self  # TODO (JDH): Should we return a copy instead?
            case int():
                return self._index_length_int(indices)
            case slice():
                return self._index_length_slice(indices)
            case list() | tuple():
                return self._index_length_list(indices)
            case torch.Tensor() if indices.ndim == 1:
                return self._index_length_1d_tensor(indices)
            case torch.Tensor() if indices.ndim > 1:
                raise NotImplementedError("Indexing length with multidimensional torch tensors is not yet implemented.")
                # if indices.dtype == torch.bool:
                #     # BoolTensor that has flattened the length dim along with other dims
                #     # TODO (JDH): Set lengths 1 for any examples that have at least one True value.
                #     pass
                # else:
                #     # IntTensor that creates new dims off of the length dim
                #     pass
            case _:
                raise NotImplementedError(f"Indexing with {indices} is not supported.")

    def _index_length_int(self, index: int) -> "StreamMetadata":
        # Convert negative indices to positive
        if index < 0:
            index = self.max_length + index

        # Set lengths to 1 for all examples except those where the integer index is in padding (set to 0)
        lengths = (self.lengths - index).clamp(min=0, max=1)
        sos = self.sos.clone() if index == 0 else torch.zeros_like(self.sos)
        eos = torch.from_numpy(update_eos_from_integer(self.eos.numpy(), self.lengths.numpy(), index))
        return StreamMetadata(deepcopy(self.ids), sos, eos, lengths)

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

        sos = self.sos.clone() if start == 0 else torch.zeros_like(self.sos)
        eos = torch.from_numpy(update_eos_from_integer(self.eos.numpy(), self.lengths.numpy(), stop - 1))
        return StreamMetadata(deepcopy(self.ids), sos, eos, lengths)

    def _index_length_list(self, indices: Union[List[int], Tuple[int]]) -> "StreamMetadata":
        # Convert to numpy arrays for faster manipulation and numba jit support.
        lengths_np = self.lengths.numpy()  # 2 µs
        indices_np = np.array(indices)  # 2 µs
        make_indices_positive_(indices_np, self.max_length)  # inplace

        # Raise error if the indices are not sorted
        if not is_sorted_ascending(indices_np):
            raise RuntimeError("Indices must be sorted when indexing length with lists or tuples.")

        # Update lengths, sos, and eos
        min_i, max_i = minmax(indices_np)
        lengths = update_lengths_from_list_of_indices(lengths_np, indices_np)
        sos = self.sos.clone() if min_i == 0 else torch.zeros_like(self.sos)
        
        eos = torch.from_numpy(update_eos_from_integer(self.eos.numpy(), lengths_np, max_i - 1))  # 2 µs
        return StreamMetadata(deepcopy(self.ids), sos, eos, lengths)

    def _index_length_1d_tensor(self, indices: torch.Tensor) -> "StreamMetadata":
        if indices.dtype == torch.bool:
            return self._index_length_1d_booltensor(indices)
        return self._index_length_1d_inttensor(indices)
    
    def _index_length_1d_booltensor(self, indices: torch.BoolTensor) -> "StreamMetadata":
        return self._index_length_list(indices.nonzero().squeeze().tolist())  # Adds ~10 µs

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

        if len(metas) == 1:
            return deepcopy(metas[0])

        assert all(isinstance(s, StreamMetadata) for s in metas)
        ids = list(itertools.chain.from_iterable([s.ids for s in metas]))
        sos = torch.cat([s.sos for s in metas], dim=0)
        eos = torch.cat([s.eos for s in metas], dim=0)
        lengths = torch.cat([s.lengths for s in metas], dim=0)
        return cls(ids, sos, eos, lengths)

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
        return cls(ids, sos, eos, lengths)

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

        split_first = self.sos.split(split_size_or_sections)
        split_last = self.eos.split(split_size_or_sections)
        split_lengths = self.lengths.split(split_size_or_sections)
        args_iter = zip(split_ids, split_first, split_last, split_lengths)

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

        # TODO: Something like this should be the standard for slicing the StreamMetadata object
        # TODO (JDH): Probably use .index_length() with a slice.
        def submeta(i, j):
            lengths = (self.lengths - i).clip(min=0, max=j - i)
            sos = self.sos.clone() if (j > 0) and (i == 0) else torch.zeros_like(self.sos)
            eos = (i < self.lengths) & (self.lengths <= j)
            ids = deepcopy(self.ids)
            return stream_metadata(ids, sos, eos, lengths)

        return [submeta(i, j) for i, j in zip(start, end)]

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
        return self.__copy__()

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
) -> StreamMetadata:
    """Create a StreamMetadata object from the given arguments.

    Args:
        ids (Union[str, List[str]]): The ids of the input tensors.
        sos (bool): Whether the input tensors are the first in a batch.
        eos (bool): Whether the input tensors are the last in a batch.
        lengths (Union[int, List[int]]): The lengths of the input tensors.
        chunk_index (int): The index of the chunk in the batch.
        num_chunks (Optional[int], optional): The number of chunks in the batch. Defaults to None.

    Returns:
        StreamMetadata: The created StreamMetadata object.
    """
    return StreamMetadata(
        ids=ids,
        sos=sos,
        eos=eos,
        lengths=lengths,
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
            f"Function {func.__name__} is not handled by StreamTensor.__torch_function__ and may not work as expected."
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
    data, meta: StreamMetadata, names: Tuple[Union[None, int]], dtype: torch.dtype = None, device: torch.device = None
) -> StreamTensor:
    """Convert a tensor to a StreamTensor. See also `torch.as_tensor`."""
    data = torch.as_tensor(data, dtype=dtype, device=device)
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
