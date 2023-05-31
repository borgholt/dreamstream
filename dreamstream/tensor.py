import functools
import uuid
import itertools

from copy import deepcopy
from typing import Any, Callable, List, Self, Tuple, Sequence, Optional, Union

import torch
import numpy as np

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from dreamstream.utils.flags import BATCH, LENGTH


STREAM_TENSOR_FUNCTIONS = dict()



# TODO (JDH): Make StreamState methods like cat, split and index lazily evaluated such that they only evaluate when they
# are needed. This minimizes overhead computation on StreamTensors that end up as leaf nodes in the computation graph.


class StreamState:
    """State associated with a batch of streamed input tensors."""

    ids: List[str]
    is_first: torch.BoolTensor
    is_last: torch.BoolTensor
    lengths: torch.IntTensor

    def __init__(
        self,
        ids: Union[str, List[str]],
        is_first: Union[bool, List[bool], torch.BoolTensor],
        is_last: Union[bool, List[bool], torch.BoolTensor],
        lengths: Union[int, List[int], torch.IntTensor],
    ):
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(lengths, int):
            lengths = [lengths]
        if isinstance(is_first, bool):
            is_first = [is_first]
        if isinstance(is_last, bool):
            is_last = [is_last]

        if not len(ids) == len(lengths) == len(is_first) == len(is_last):
            raise ValueError("ids, lengths, is_first and is_last must have the same length.")

        is_first = torch.as_tensor(is_first, dtype=torch.bool)
        is_last = torch.as_tensor(is_last, dtype=torch.bool)
        lengths = torch.as_tensor(lengths, dtype=torch.int)

        if not all(isinstance(i, str) for i in ids):
            raise ValueError("ids must be a list of strings.")

        if is_first.ndim > 1 or is_last.ndim > 1 or lengths.ndim > 1:
            raise ValueError("is_first, is_last and lengths must be 1-dimensional.")

        self.ids = ids
        self._is_first = is_first
        self._is_last = is_last
        self._lengths = lengths
        
        self._max_length = None
        self._min_length = None
        self._lengths_updated = True
        self._update_min_max_length()
        
        self._any_first = None
        self._any_last = None
        self._all_first = None
        self._all_last = None
        self._any_first_or_last = None
        self._all_first_and_last = None
        self._first_last_updated = True
        self._self_update_logical()
        
    def __getitem__(self, indices: Union[int, slice, List[Any], Tuple[Any, ...], torch.IntTensor, torch.BoolTensor]):
        """Index the state along the batch and/or length dimensions."""
        raise NotImplementedError()

    def index_batch(self, indices: Union[int, slice, List[int], Tuple[int, ...], torch.IntTensor, torch.BoolTensor]) -> "StreamState":
        """Return a StreamState object with the specified batch indices."""
        if isinstance(indices, torch.BoolTensor):
            ids = [id for i, id in enumerate(self.ids) if indices[i]]
            is_first = self.is_first[indices]
            is_last = self.is_last[indices]
            lengths = self.lengths[indices]
            return StreamState(ids, is_first, is_last, lengths)

        if isinstance(indices, int):
            ids = [self.ids[indices]]
            is_first = self.is_first[[indices]]
            is_last = self.is_last[[indices]]
            lengths = self.lengths[[indices]]
            return StreamState(ids, is_first, is_last, lengths)

        if isinstance(indices, slice):
            ids = self.ids[indices]
        else:  # List[int], Tuple[int, ...], torch.IntTensor
            ids = [self.ids[i] for i in indices]

        is_first = self.is_first[indices]
        is_last = self.is_last[indices]
        lengths = self.lengths[indices]
        return StreamState(ids, is_first, is_last, lengths)

    def index_length(self, indices: Union[int, slice, List[int], Tuple[int], torch.IntTensor, torch.BoolTensor]) -> "StreamState":
        """Return a StreamState object with the specified length indices."""
        if isinstance(indices, int):
            # Convert negative indices to positive
            if indices < 0:
                indices = self.max_length + indices
            # Set lengths to 1 for all examples except those where the integer index is in padding.
            lengths = (self.lengths - indices).clamp(max=1)
            is_first = self.is_first.clone() if indices == 0 else torch.zeros_like(self.is_first)
            is_last = self.is_last & (self.lengths <= indices + 1)  # TODO (JDH): Would `self.lengths < indices` be better?
            return StreamState(deepcopy(self.ids), is_first, is_last, lengths)

        if isinstance(indices, slice):
            # 
            start, stop, stride = indices.indices(self.max_length)
            if stride < 0:
                # TODO (JDH): Allow negative strides only if all examples have the same length equal to tensor's length.
                msg = "Negative strides are not supported on StreamTensors. Instead, use `reverse_sequence`."
                raise NotImplementedError(msg)

            if stride == 1:
                lengths = (self.lengths.clamp(max=stop) - start)
            else:
                # Compute the right-most length index that is included in the slice from slice(start, stop, stride)
                stop = self.max_length - ((self.max_length - 1 - start) % stride)
                lengths = (self.lengths.clamp(max=stop) - start).div(stride).ceil().to(self.lengths.dtype)

            is_first = self.is_first.clone() if start == 0 else torch.zeros_like(self.is_first)
            is_last = self.is_last & (self.lengths <= stop)  # TODO (JDH): Would `self.lengths < indices` be better?
            return StreamState(deepcopy(self.ids), is_first, is_last, lengths)

        if isinstance(indices, (list, tuple)):
            raise NotImplementedError("Indexing with lists and tuples is not yet implemented.")
            # if not all(isinstance(i, int) for i in indices):
            #     raise ValueError("Indices must contain integers but looked like multidimensional indexing.")

            # Convert negative indices to positive
            indices = [i if i >= 0 else self.max_length + i for i in indices]
            
            # Raise error if the indices are not sorted and any index is into padding (i.e. lengths < indices)
            # TODO (JDH)

            # if 0 in indices and indices.index(0) != 0:
            #     raise NotImplementedError("Cannot index StreamState with 0 and other indices.")

        if isinstance(indices, torch.Tensor) and indices.ndim == 1:
            raise NotImplementedError("Indexing with 1D torch tensors is not yet implemented.")

        if isinstance(indices, torch.Tensor) and indices.ndim > 1:
            raise NotImplementedError("Indexing with multidimensional torch tensors is not supported.")
            if indices.dtype == torch.bool:
                # BoolTensor that has flattened the length dim along with other dims
                # TODO (JDH): Set lengths 1 for any examples that have at least one True value.
                pass
            else:
                # IntTensor that creates new dims off of the length dim
                pass

        raise NotImplementedError(f"Indexing with {indices} is not supported.")

    def __copy__(self):
        """Return a deep copy of the StreamState object."""
        return StreamState(
            ids=deepcopy(self.ids),
            is_first=self.is_first.clone(),
            is_last=self.is_last.clone(),
            lengths=self.lengths.clone(),
        )

    def clone(self):
        """Return a deep copy of the StreamState object."""
        return self.__copy__()

    def __len__(self):
        return len(self.ids)
    
    def _self_update_logical(self):
        if self._first_last_updated:
            self._any_first = self.is_first.any().item()
            self._any_last = self.is_last.any().item()
            self._all_first = self.is_first.all().item()
            self._all_last = self.is_last.all().item()
            self._any_first_or_last = self._any_first or self._any_last
            self._all_first_and_last = self._all_first and self._all_last
            self._first_last_updated = False
            
    def _update_min_max_length(self):
        if self._lengths_updated:
            self._max_length = self.lengths.max().item()
            self._min_length = self.lengths.min().item()
            self._lengths_updated = False
        
    
    @property
    def any_first(self):
        self._self_update_logical()
        return self._any_first
    
    @property
    def any_last(self):
        self._self_update_logical()
        return self._any_last
    
    @property
    def all_first(self):
        self._self_update_logical()
        return self._all_first
    
    @property
    def all_last(self):
        self._self_update_logical()
        return self._all_last
    
    @property
    def any_first_or_last(self):
        self._self_update_logical()
        return self._any_first_or_last
    
    @property
    def all_first_and_last(self):
        self._self_update_logical()
        return self._all_first_and_last
    
    @property
    def is_first(self):
        return self._is_first
    
    @property
    def is_last(self):
        return self._is_last
    
    @is_first.setter
    def is_first(self, value):
        self._is_first = value
        assert self._is_first.dtype == torch.bool
        self._first_last_updated = True
        
    @is_last.setter
    def is_last(self, value):
        self._is_last = value
        assert self._is_first.dtype == torch.bool
        self._first_last_updated = True
        
    def __eq__(self, other):        
        return (
            self.ids == other.ids
            and self.is_first.equal(other.is_first)
            and self.is_last.equal(other.is_last)
            and self.lengths.equal(other.lengths)
        )
    
    @property
    def lengths(self):
        return self._lengths

    @property
    def max_length(self):
        """Return the maximum length of the batch and recompute only if lengths have been mutated."""
        self._update_min_max_length()
        return self._max_length
    
    @property
    def min_length(self):
        """Return the maximum length of the batch and recompute only if lengths have been mutated."""
        self._update_min_max_length()
        return self._min_length

    @property
    def _first_lengths(self):
        return self.lengths[self.is_first]

    @property
    def _last_lengths(self):
        return self.lengths[self.is_last]

    @lengths.setter
    def lengths(self, i):
        self._lengths = i
        self._lengths_updated = True
        
    @max_length.setter
    def max_length(self, i):
        raise AttributeError("max_length is read-only.")
    
    @min_length.setter
    def min_length(self, i):
        raise AttributeError("min_length is read-only.")
    
    @_first_lengths.setter
    def _first_lengths(self, i):
        self._lengths[self.is_first] = i
        self._lengths_updated = True
    
    @_last_lengths.setter
    def _last_lengths(self, i):
        self._lengths[self.is_last] = i
        self._lengths_updated = True

    def size(self):
        return len(self)

    def drop_empty(self) -> Self:
        """Drop any tensors that are empty."""
        return self.index_batch(self.lengths > 0)

    # def filter(self, mask: torch.BoolTensor):
    #     """Keep only the batch examples where the mask is True."""
    #     if not mask.all():
    #         self.is_first = self.is_first[mask]
    #         self.is_last = self.is_last[mask]
    #         self.lengths = self.lengths[mask]
    #         self.ids = [id for i, id in enumerate(self.ids) if mask[i]]

    def __repr__(self):
        return f"StreamState(size={self.size()})"

    @classmethod
    def cat_batch(cls, stream_states: List["StreamState"]) -> "StreamState":
        """Concatenate a list of StreamState objects along the batch dimension.

        Args:
            stream_states (List[StreamState]): The StreamState objects to concatenate.

        Returns:
            StreamState: The concatenated StreamState object.
        """
        
        if len(stream_states) == 1:
            return deepcopy(stream_states[0])
        
        assert all(isinstance(s, StreamState) for s in stream_states)
        ids = list(itertools.chain.from_iterable([s.ids for s in stream_states]))
        is_first = torch.cat([s.is_first for s in stream_states], dim=0)
        is_last = torch.cat([s.is_last for s in stream_states], dim=0)
        lengths = torch.cat([s.lengths for s in stream_states], dim=0)
        return cls(ids, is_first, is_last, lengths)
    
    @classmethod
    def cat_length(cls, stream_states: List["StreamState"]) -> "StreamState":
        """Concatenate a list of StreamState objects along the length dimension.

        Args:
            stream_states (List[StreamState]): The StreamState objects to concatenate.

        Returns:
            StreamState: The concatenated StreamState object.
        """
        
        if len(stream_states) == 1:
            return deepcopy(stream_states[0])
        
        j = len(stream_states) - 1
        for i, s in enumerate(stream_states):
            if i != 0 and s.ids != stream_states[0].ids:
                raise ValueError("Cannot concatenate StreamStates with different ids.")
            if i != 0 and s.any_first:
                raise ValueError('StreamStates where any "is_first" is True should be first when concatenated.')
            if i != j and s.any_last:
                raise ValueError('StreamStates where any "is_last" is True should be last when concatenated.')
        
        ids = deepcopy(stream_states[0].ids)
        is_first = stream_states[0].is_first.clone()
        is_last = stream_states[-1].is_last.clone()
        lengths = sum([s.lengths for s in stream_states])
        return cls(ids, is_first, is_last, lengths)
    
    @classmethod
    def cat(cls, stream_states: List["StreamState"], dim: str) -> "StreamState":
        """Concatenate a list of StreamState objects along a given dimension."""   
        if dim == LENGTH:
            return cls.cat_length(stream_states)
        elif dim == BATCH:
            return cls.cat_batch(stream_states)
        else:
            raise ValueError(f"Invalid dimension: {dim}")
    
    def split_batch(self, split_size_or_sections: Union[int, List[int]]) -> List["StreamState"]:
        """Split a StreamState object into a list of StreamState objects along the batch dimension.

        Args:
            split_size_or_sections (Union[int, List[int]]): Size of a single chunk or list of sizes for each chunk.

        Returns:
            List[StreamState]: The split StreamState objects.
        """
        if isinstance(split_size_or_sections, list) and sum(split_size_or_sections) != len(self):
            raise ValueError("Sum of split sizes must equal the size of the StreamState object.")

        if isinstance(split_size_or_sections, int):
            start = range(0, len(self), split_size_or_sections)
            split_ids = [self.ids[i : i + split_size_or_sections] for i in start]
        else:
            slices = np.cumsum([0] + split_size_or_sections)
            split_ids = [self.ids[i:j] for i, j in zip(slices[:-1], slices[1:])]

        split_first = self.is_first.split(split_size_or_sections)
        split_last = self.is_last.split(split_size_or_sections)
        split_lengths = self.lengths.split(split_size_or_sections)
        args_iter = zip(split_ids, split_first, split_last, split_lengths)

        return [stream_state(*args) for args in args_iter]

    def split_length(self, split_size_or_sections: Union[int, List[int]]) -> List["StreamState"]:
        """Split a StreamState object into a list of StreamState objects along the length dimension.

        Args:
            split_size_or_sections (Union[int, List[int]]): Size of a single chunk or list of sizes for each chunk.

        Returns:
            List[StreamState]: The split StreamState objects.
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

        # TODO: Something like this should be the standard for slicing the StreamState object
        def substate(i, j):
            lengths = (self.lengths - i).clip(min=0, max=j - i)
            is_first = self.is_first.clone() if (j > 0) and (i == 0) else torch.zeros_like(self.is_first)
            is_last = (i < self.lengths) & (self.lengths <= j)
            ids = deepcopy(self.ids)
            return stream_state(ids, is_first, is_last, lengths)

        return [substate(i, j) for i, j in zip(start, end)]

    def split(self, split_size_or_sections: Union[int, List[int]], dim: str) -> List["StreamState"]:
        """Split a StreamState object into a list of StreamState objects along a given dimension."""
        if dim == LENGTH:
            return self.split_length(split_size_or_sections)
        elif dim == BATCH:
            return self.split_batch(split_size_or_sections)
        raise ValueError(f"Invalid dimension: {dim}")

    def unbind_batch(self) -> List["StreamState"]:
        """Split a StreamState object into a list of StreamState objects along the batch dimension.

        Returns:
            List[StreamState]: The split StreamState objects.
        """
        return self.split_batch(1)


# TODO (JDH): Implement a MultiLengthStreamState class that can handle multiple length dimensions.
class MultiLengthStreamState():
    def __init__(self) -> None:
        raise NotImplementedError()


def stream_state(
    ids: Union[str, List[str]],
    is_first: Union[bool, List[bool], torch.BoolTensor],
    is_last: Union[bool, List[bool], torch.BoolTensor],
    lengths: Union[int, List[int], torch.IntTensor],
) -> StreamState:
    """Create a StreamState object from the given arguments.

    Args:
        ids (Union[str, List[str]]): The ids of the input tensors.
        is_first (bool): Whether the input tensors are the first in a batch.
        is_last (bool): Whether the input tensors are the last in a batch.
        lengths (Union[int, List[int]]): The lengths of the input tensors.
        chunk_index (int): The index of the chunk in the batch.
        num_chunks (Optional[int], optional): The number of chunks in the batch. Defaults to None.

    Returns:
        StreamState: The created StreamState object.
    """
    return StreamState(
        ids=ids,
        is_first=is_first,
        is_last=is_last,
        lengths=lengths,
    )


class StreamTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, stream_state: StreamState, *args, **kwargs) -> torch.Tensor:
        """Return a new StreamTensor object (cls is StreamTensor)."""
        # print(f"StreamTensor.__new__ called with data: {data}, stream_state: {stream_state}")
        return super().__new__(cls, data, *args, **kwargs)  # call torch.Tensor.__new__

    def __init__(self, data, stream_state: StreamState, *args, names: List[str] = None, **kwargs):
        """Initialize a StreamTensor object (self is StreamTensor, data is e.g. torch.Tensor)."""
        # TODO (JDH): Check that either BATCH or LENGTH are in either data.names or names and raise error if not.
        # print(f"StreamTensor.__init__ called with data: {data}, stream_state: {stream_state}")
        super().__init__()
        self._stream_state = stream_state

    @classmethod
    def __torch_function__(cls, func: Callable, types: List[torch.Tensor], args=(), kwargs=None):
        # TODO (JDH): Deal with empty StreamTensors?
        # import IPython
        # IPython.embed(using=False, header="Hello from StreamTensor.__torch_function__")
        print("TEST", func.__name__)
        if kwargs is None:
            kwargs = dict()

        is_handled = func in STREAM_TENSOR_FUNCTIONS and all(issubclass(t, (torch.Tensor, StreamTensor)) for t in types)
        print(f"\n\n{func.__name__}: {is_handled=}\n\n")

        if is_handled:
            return STREAM_TENSOR_FUNCTIONS[func](*args, **kwargs)

        out = super().__torch_function__(func, types, args, kwargs)
        if isinstance(out, torch.Tensor):
            # TODO: If more than one, assert that stream_states are identical, or raise error
            stream_states = [x.stream_state for x in [*args, *kwargs.values()] if isinstance(x, StreamTensor)]
            out = StreamTensor(out, stream_state=stream_states[0])

        return out

    def set_empty(self):
        self.stream_state.set_empty()
        self.data = None

    @property
    def is_empty(self):
        return self.size(LENGTH) == 0

    @property
    def stream_state(self) -> StreamState:
        return self._stream_state

    @stream_state.setter
    def stream_state(self, s: StreamState):
        self._stream_state = s

    @property
    def batch_dim(self) -> Union[int, None]:
        try:
            return self.names.index(BATCH)
        except ValueError:
            return None

    @property
    def length_dim(self) -> Union[int, None]:
        try:
            return self.names.index(LENGTH)
        except ValueError:
            return None

    def _is_batch_dim(self, dim: int) -> bool:
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
        tensor = torch.Tensor(self)
        if not keep_names:
            tensor.rename_(None)
        return tensor
    
    def detach_stream(self) -> Tuple[torch.Tensor, StreamState, Tuple[str]]:
        names = self.names
        state = self.stream_state
        tensor = self.tensor()
        return tensor, state, names
    
    def drop_empty(self) -> Self:
        """Remove empty tensors from the batch."""
        if self.stream_state.min_length > 0:
            return self
        if self.stream_state.max_length == 0:
            return None
        if len(self.stream_state) == 1 and self.stream_state.max_length > 0:
            return self
        tensor, state, names = self.detach_stream()
        batch_dim = names.index(BATCH)
        tensor = torch.index_select(tensor, batch_dim, state.lengths.nonzero().squeeze())
        return as_stream_tensor(data=tensor, state=state.drop_empty(), names=names)

    def named_tensor(self) -> torch.Tensor:
        """Return the underlying torch.Tensor with names."""
        return self.tensor(keep_names=True)
    
    def unpad_sequence(self) -> torch.Tensor:
        """Remove padding along the specified dimension."""
        batch_dim = self.names.index(BATCH)
        length_dim = self.names.index(LENGTH)
        if batch_dim < length_dim:
            length_dim -= 1
        return [x.narrow(length_dim, 0, x.stream_state.lengths.item()) for x in self.unbind(dim=batch_dim)]

    def iter_chunks(self):
        """Return an iterator over chunks of the tensor."""
        # TODO (LB): Implement this.
        raise NotImplementedError()

    def to_chunks(self):
        """Return a list of chunks of the tensor."""
        # TODO (LB): Implement this. Should return a ChunkedList object.
        raise NotImplementedError()


def as_stream_tensor(
    data, state: StreamState, names: Tuple[Union[None, int]], dtype: torch.dtype = None, device: torch.device = None
) -> StreamTensor:
    data = torch.as_tensor(data, dtype=dtype, device=device)
    data = data.refine_names(*names)  # Make the tensor named if it isn't already.
    return StreamTensor(data=data, stream_state=state)


def stream_tensor(
    data,
    state: StreamState,
    names: Tuple[Union[None, int]],
    dtype: torch.dtype = None,
    device: torch.device = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> StreamTensor:
    if isinstance(data, torch.Tensor) and data.names != names:
        data = data.rename(*names)

    data = torch.tensor(
        data, names=names, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory
    )
    return StreamTensor(data=data, stream_state=state)
