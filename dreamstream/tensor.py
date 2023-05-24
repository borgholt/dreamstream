import uuid
import math
import itertools
import functools
from copy import deepcopy
from typing import Callable, List, Tuple, Sequence, Optional, Union

import torch
import numpy as np
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from dreamstream.utils.flags import BATCH, LENGTH


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
        self._lengths_updated = False
        self._max_length = lengths.max().item()
        
        self._any_first = None
        self._any_last = None
        self._all_first = None
        self._all_last = None
        self._any_first_or_last = None
        self._all_first_and_last = None
        self._any_mutated = True
        self._self_update_logical()

    def __len__(self):
        return len(self.ids)
    
    def _self_update_logical(self):
        if self._any_mutated:
            self._any_first = self.is_first.any().item()
            self._any_last = self.is_last.any().item()
            self._all_first = self.is_first.all().item()
            self._all_last = self.is_last.all().item()
            self._any_first_or_last = self._any_first or self._any_last
            self._all_first_and_last = self._all_first and self._all_last
            self._any_mutated = False
    
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
        self._any_mutated = True
        
    @is_last.setter
    def is_last(self, value):
        self._is_last = value
        assert self._is_first.dtype == torch.bool
        self._any_mutated = True
        
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
        if self._lengths_updated:
            self._max_length = self._lengths.max().item()
            self._lengths_updated = False
        return self._max_length

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

    def drop_empty(self):
        """Drop any tensors that are empty."""
        self.filter(self.lengths > 0)

    def filter(self, mask: torch.BoolTensor):
        """Keep only the batch examples where the mask is True."""
        if not mask.all():
            self.is_first = self.is_first[mask]
            self.is_last = self.is_last[mask]
            self.lengths = self.lengths[mask]
            self.ids = [id for i, id in enumerate(self.ids) if mask[i]]

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
            split_ids = [self.ids[i:i+split_size_or_sections] for i in start]
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
            lengths = (self.lengths - i).clip(min=0, max=j-i)
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
        else:
            raise ValueError(f"Invalid dimension: {dim}")
        
    


    
    def unbind_batch(self) -> List["StreamState"]:
        """Split a StreamState object into a list of StreamState objects along the batch dimension.

        Returns:
            List[StreamState]: The split StreamState objects.
        """
        return self.split_batch(1)


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


STREAM_TENSOR_FUNCTIONS = dict()


# TODO (JDH): I think this should not inherit from torch.Tensor, but instead have a torch.Tensor as an attribute.
#             The problem is that calls to torch.cat, torch.stack, etc. become recursive.

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

    def set_empty(self):
        self.stream_state.set_empty()
        self.data = None

    @property
    def is_empty(self):
        return self.size(LENGTH) == 0

    @classmethod
    def __torch_function__(cls, func: Callable, types: List[torch.Tensor], args=(), kwargs=None):
        # print("StreamTensor.__torch_function__ called with func:", func)
        # TODO (JDH): If func is cat, stack, vstack or hstack along batch dim, apply this also to stream_states.
        if kwargs is None:
            kwargs = dict()

        is_handled = func in STREAM_TENSOR_FUNCTIONS and all(issubclass(t, (torch.Tensor, StreamTensor)) for t in types)
        #print(f"\n\n{func.__name__}: {is_handled}\n\n")
        if is_handled:
            #print("\n\n", cls, func, types, args, kwargs, "\n\n")
            return STREAM_TENSOR_FUNCTIONS[func](*args, **kwargs)

        out = super().__torch_function__(func, types, args, kwargs)

        if isinstance(out, torch.Tensor):
            # TODO: If more than one, assert that stream_states are identical, or raise error
            stream_states = [x.stream_state for x in [*args, *kwargs.values()] if isinstance(x, StreamTensor)]
            out = StreamTensor(out, stream_state=stream_states[0])

        return out

    @property
    def stream_state(self) -> StreamState:
        return self._stream_state

    @stream_state.setter
    def stream_state(self, s: StreamState):
        self._stream_state = s

    def _is_batch_dim(self, dim: int) -> bool:
        return self.names[dim] == BATCH

    def is_length_dim(self, dim: int) -> bool:
        # TODO: Refine to include multiple length dims
        return self.names[dim] == LENGTH
    
    def max_length(self):
        # TODO: Refine to include multiple length dims
        return self.size(self.names.index(LENGTH))
    
    def batch_size(self):
        return self.size(self.names.index(BATCH))

    def tensor(self, keep_names=False) -> torch.Tensor:
        """Return the underlying torch.Tensor. Names are removed by defa"""
        if not keep_names:
            self.rename_(None)
        return torch.Tensor(self)

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


def stream_tensor(
    data,
    state: List[StreamState] = None,
    names: List[str] = None,
    ids: Union[str, List[str]] = None,
    is_first: Union[bool, List[bool], torch.BoolTensor] = None,
    is_last: Union[bool, List[bool], torch.BoolTensor] = None,
    lengths: Union[int, List[int], torch.IntTensor] = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
    requires_grad: bool = False,
    pin_memory: bool = False
) -> StreamTensor:
    """Create a StreamTensor object from the given arguments.

    This method will convert data to a `torch.Tensor` if it is not already a `torch.Tensor`. It will use 
    `torch.as_tensor` for this to avoid unnecessary copying, unless kwargs contains "requires_grad" or "pin_memory", 
    in which case we are forced to make a copy by using `torch.tensor`.

    Args:
        data (torch.Tensor): The input tensor.
        ids (Union[str, List[str]]): The ids of the input tensors.
        is_first (bool): Whether the input tensors are the first in a batch.
        is_last (bool): Whether the input tensors are the last in a batch.
        lengths (Union[int, List[int]]): The lengths of the input tensors.
        chunk_index (int): The index of the chunk in the batch.
        num_chunks (Optional[int], optional): The number of chunks in the batch. Defaults to None.

    Returns:
        StreamTensor: The constructed StreamTensor object.
    """
    
    # Check that we can create a named tensor.
    if names is None:
        if not isinstance(data, torch.Tensor) or (LENGTH not in data.names):
            raise ValueError("Must provide `names` either via tensor data or names argument.")
    else:
        if (LENGTH not in names):
            raise ValueError("Must provide `names` either via tensor data or names argument.")

    # Check if we need to copy the data.
    requires_copy = False
    if requires_grad or pin_memory:
        # If requires_grad or pin_memory is True, we may need to make a copy of the data, but only if data is not
        # already a tensor with the requested requires_grad or pin_memory attributes.
        data_requires_grad = data.requires_grad if isinstance(data, torch.Tensor) else False
        data_pin_memory = data.is_pinned() if isinstance(data, torch.Tensor) else False
        requires_copy = (requires_grad and not data_requires_grad) or (pin_memory and not data_pin_memory)

    # TODO: Shouldn't we just have both a stream_tensor and as_stream_tensor function instead of this?
    if requires_copy:
        # If requires_copy is True, we need to make a copy of the data, so we use torch.tensor.
        data = torch.tensor(data, names=names, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)
    else:
        # If data is already a tensor with the requested dtype and device then data itself is returned, but if data is 
        # a tensor with a different dtype or device then it’s copied as if using data.to(dtype=dtype, device=device).
        data = torch.as_tensor(data, dtype=dtype, device=device)
        if names is not None:
            data = data.refine_names(*names)  # Make the tensor named if it isn't already.

    assert BATCH in data.names and LENGTH in data.names, "data must have batch and length dimensions, 'B' and 'L'."

    if state is None:
        state = stream_state(ids, is_first, is_last, lengths)

    s = StreamTensor(data=data, stream_state=state)
    return s

def as_stream_tensor():
    raise NotImplementedError()

def all_stream_tensors(tensors: List[Union[StreamTensor, Tensor]]) -> bool:
    """Return True if all tensors are StreamTensors."""
    return all(isinstance(t, StreamTensor) for t in tensors)


def require_all_stream_tensors(tensors: List[Union[StreamTensor, Tensor]], message: str = None):
    """Raise an error if any of the tensors are not StreamTensors."""
    if not all_stream_tensors(tensors):
        message = message or "All tensors must be StreamTensors."
        raise ValueError(message)


def implements(torch_function):
    """Register a torch function override for StreamTensor."""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        STREAM_TENSOR_FUNCTIONS[torch_function] = func
        return func

    return decorator



        


# @implements(torch.stack)
# def stack(tensors: List[Union[StreamTensor, Tensor]], dim=0, *, out=None):
#     """Create a new dim and name it """


# @implements(torch.vstack)
# @implements(torch.hstack)

# @implements(torch.split)
# @implements(torch.chunk)

# @implements(torch.flatten)

# @implements(torch.squeeze)  # Never remove batch or length dims
# @implements(torch.unsqueeze)  # Give new dim default name

# @implements(torch.nn.utils.rnn.pad_sequence)
# @implements(torch.nn.utils.rnn.unpad_sequence)

# reduction methods
# @implements(torch.sum)  # Never remove batch or length dims
# @implements(torch.mean)
# @implements(torch.std)

# indexing, slicing, joining, mutating methods
# @implements(torch.index)
# @implements(torch.gather)
# @implements(torch.scatter)
# @implements(torch.gather_index)


# @implements(torch.transpose)
# @implements(torch.permute)


    

    
    
    
    


        
        
        
        
    

if __name__ == "__main__":
    x = torch.randn(32, 128, 56).tolist()
    ids = [uuid.uuid4().hex for _ in range(32)]
    names = [BATCH, LENGTH, "F"]
    is_first = torch.tensor([True, True, True] + [False] * 29)
    is_last = torch.tensor([False] * 29 + [True, True, True])
    lengths = torch.randint(20, 56, (32,))

    s = stream_tensor(
        x,
        ids=ids,
        is_first=is_first,
        is_last=is_last,
        lengths=lengths,
        dtype=torch.float32,
        device="cpu",
        requires_grad=False,
        pin_memory=False,
        names=names,
    )
    
    # c1 = torch.cat([s, s], dim=0)
    
    # a = torch.randn(32, 128, 56)
    
    p1 = torch.permute(s, (2, 0, 1)) # WORKS
    # c2 = torch.cat([s, a], dim=0)

    # from random import randint
    # tensors = [torch.rand(256, randint(50, 100)) for _ in range(4)]
    # ids = [uuid.uuid4().hex for _ in range(4)]
    # l = LongformList(tensors=tensors, ids=ids, chunk_size=20, names=("F", LENGTH))