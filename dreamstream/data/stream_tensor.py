import itertools

from typing import Callable, List, Tuple, Optional, Union

import functools
import uuid

import torch

from torch import Tensor


BATCH = "B"
LENGTH = "L"


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
        self.is_first = is_first
        self.is_last = is_last
        self.lengths = lengths
        
        self._any_first = self.is_first.any().item()
        self._any_last = self.is_last.any().item()
        
        self._all_first = self.is_first.all().item()
        self._all_last = self.is_last.all().item()
        
        self._any_first_or_last = self._any_first or self._any_last

    def __len__(self):
        return len(self.ids)

    @property
    def _first_lengths(self):
        return self.lengths[self.is_first]
    
    @property
    def _last_lengths(self):
        return self.lengths[self.is_last]
    
    @_first_lengths.setter
    def _first_lengths(self, i):
        self.lengths[self.is_first] = i
    
    @_last_lengths.setter
    def _last_lengths(self, i):
        self.lengths[self.is_last] = i


    def size(self):
        return len(self)

    def drop_empty(self):
        """Drop any tensors that are empty."""
        self.filter(self.lengths > 0)

    def filter(self, mask: torch.BoolTensor):
        """Keep only the batch examples where the mask is True."""
        self.is_first = self.is_first[mask]
        self.is_last = self.is_last[mask]
        self.lengths = self.lengths[mask]
        self.ids = [id for i, id in enumerate(self.ids) if mask[i]]

    def __repr__(self):
        return f"StreamState(size={self.size()})"

    @classmethod
    def cat(cls, stream_states: List["StreamState"]) -> "StreamState":
        """Concatenate a list of StreamState objects along the batch dimension.

        Args:
            stream_states (List[StreamState]): The StreamState objects to concatenate.

        Returns:
            StreamState: The concatenated StreamState object.
        """
        assert all(isinstance(s, StreamState) for s in stream_states)
        ids = list(itertools.chain.from_iterable([s.ids for s in stream_states]))
        is_first = torch.cat([s.is_first for s in stream_states], dim=0)
        is_last = torch.cat([s.is_last for s in stream_states], dim=0)
        lengths = torch.cat([s.lengths for s in stream_states], dim=0)
        return cls(ids, is_first, is_last, lengths)


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
        if is_handled:
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
        return self.names[dim] == LENGTH

    def tensor(self) -> torch.Tensor:
        """Return the underlying torch.Tensor."""
        # TODO (JDH): Verify that this is OK.
        # import IPython
        # IPython.embed(using=False)
        # tensor.__class__ = torch.Tensor
        # tensor.__torch_function__ = torch.Tensor.__torch_function__
        tensor = torch.Tensor(self)
        return tensor

    def unpad_sequence(self) -> torch.Tensor:
        """Remove padding along the specified dimension."""
        # TODO (JDH): Implement this.
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
    pin_memory: bool = False,
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
        StreamTensor: The created StreamTensor object.
    """
    
    # Check that we can create a named tensor.
    if names is None:
        if not isinstance(data, torch.Tensor) or (BATCH not in data.names) or (LENGTH not in data.names):
            raise ValueError("Must provide `names` either via tensor data or names argument.")
    else:
        if (BATCH not in names) or (LENGTH not in names):
            raise ValueError("Must provide `names` either via tensor data or names argument.")

    # Check if we need to copy the data.
    requires_copy = False
    if requires_grad or pin_memory:
        # If requires_grad or pin_memory is True, we may need to make a copy of the data, but only if data is not
        # already a tensor with the requested requires_grad or pin_memory attributes.
        data_requires_grad = data.requires_grad if isinstance(data, torch.Tensor) else False
        data_pin_memory = data.is_pinned() if isinstance(data, torch.Tensor) else False
        requires_copy = (requires_grad and not data_requires_grad) or (pin_memory and not data_pin_memory)

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


def all_stream_tensors(tensors: List[Union[StreamTensor, Tensor]]) -> bool:
    """Return True if all tensors are StreamTensors."""
    return all(isinstance(t, StreamTensor) for t in tensors)


def require_all_stream_tensors(tensors: List[Union[StreamTensor, Tensor]], message: str = None):
    """Raise an error if any of the tensors are not StreamTensors."""
    if not all_stream_tensors(tensors):
        message = message or "All tensors must be StreamTensors."
        raise ValueError(message)


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        STREAM_TENSOR_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements(torch.cat)
def cat(tensors: List[Union[StreamTensor, Tensor]], dim=0, *, out=None):
    """If dim is the batch dimension of any StreamTensor, assert all are StreamTensors and concatenate the stream 
    states as well. Else, call torch.cat.
    """
    # Concatenation of at least one StreamTensor along the batch dimension.
    is_batch_dim = [t._is_batch_dim(dim) for t in tensors if isinstance(t, StreamTensor)]
    if any(is_batch_dim):
        require_all_stream_tensors(tensors, "Cannot concatenate StreamTensor and torch.Tensor along batch dimension.")
        stream_state = StreamState.cat([t.stream_state for t in tensors])  # TODO (JDH): Make this lazily evaluated.
        # import IPython
        # IPython.embed(using=False)
        tensors = [t.tensor() for t in tensors]
        tensor = torch.cat(tensors, dim=dim, out=out)
        return StreamTensor(tensor, stream_state)

    # Concatenation of at least one StreamTensor along the length dimension.
    is_length_dim = [t.is_length_dim(dim) for t in tensors if isinstance(t, StreamTensor)]
    if any(is_length_dim):
        # Options include:
        #  1. Concatenate without padding (assume right), add lengths of all tensors to get length of concatenated.
        #  2. 1. but communicate padding information in stream_state (right, left or both)
        #  3. If only one tensor is a StreamTensor, do not update the stream_state (assuming lengths unaffected).
        raise NotImplementedError("Concatenating along the length dimension is not yet supported.")
    
    return torch.cat(tensors, dim=dim, out=out)


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

class LongformList(list):
    
    def __init__(
        self,
        tensors: Union(List[torch.Tensor], List[StreamTensor]),
        ids: List[str],
        names: Union[List[str], Tuple[str]] = None
    ):
        
        if names is not None:
            tensors = [t.refine_names(*names) for t in tensors]
        
    

if __name__ == "__main__":
    x = torch.randn(32, 128, 56).tolist()
    ids = [uuid.uuid4().hex for _ in range(32)]
    names = [BATCH, LENGTH, "feature"]
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
    
    c1 = torch.cat([s, s], dim=0)
    
    a = torch.randn(32, 128, 56)
    # c2 = torch.cat([s, a], dim=0)
    

    import IPython
    IPython.embed(using=False)
