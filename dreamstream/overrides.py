import functools
from typing import List, Union

from dreamstream.tensor import StreamTensor, StreamState, STREAM_TENSOR_FUNCTIONS
from dreamstream.utils.flags import BATCH, LENGTH

import torch
from torch import Tensor


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
        tensors = [t.tensor() for t in tensors]
        tensor = torch.cat(tensors, dim=dim, out=out)
        return StreamTensor(tensor, stream_state)

    # Concatenation of at least one StreamTensor along the length dimension.
    is_length_dim = [t.is_length_dim(dim) for t in tensors if isinstance(t, StreamTensor)]
    if any(is_length_dim):
        require_all_stream_tensors(tensors, "Cannot concatenate StreamTensor and torch.Tensor along length dimension.")
        for t in tensors:
            if t.stream_state.lengths.min() < t.max_length():
                raise NotImplementedError("Concatenating along the length dimension is only supported for non-padded tensors.")
        
        stream_state = sum([t.stream_state for t in tensors])
        tensors = [t.tensor() for t in tensors]
        tensor = torch.cat(tensors, dim=dim, out=out)
        return StreamTensor(tensor, stream_state)
            
    return torch.cat(tensors, dim=dim, out=out)


@implements(torch.permute)
def permute(tensor: StreamTensor, dims: List[int]):
    dims = [tensor.names[dim] for dim in dims]
    return tensor.align_to(*dims)

@implements(torch.Tensor.permute)
def tensor_permute(tensor: StreamTensor, *dims: int):
    return permute(tensor, dims)

# Seemingly non-overriadable functions. Used in torch.Tensor.split:
#torch._VF.split 
#torch._VF.split_with_sizes

# Overridable functions from "torch" that includes the word "split":
# split <function split at 0x7fbd4c721ea0>
# dsplit <built-in method dsplit of type object at 0x7fbe852b0540>
# hsplit <built-in method hsplit of type object at 0x7fbe852b0540>
# split <function split at 0x7fbd4c721ea0>
# split_copy <built-in method split_copy of type object at 0x7fbe852b0540>
# split_with_sizes <built-in method split_with_sizes of type object at 0x7fbe852b0540>
# split_with_sizes_copy <built-in method split_with_sizes_copy of type object at 0x7fbe852b0540>
# tensor_split <built-in method tensor_split of type object at 0x7fbe852b0540>
# unsafe_split <built-in method unsafe_split of type object at 0x7fbe852b0540>
# unsafe_split_with_sizes <built-in method unsafe_split_with_sizes of type object at 0x7fbe852b0540>
# vsplit <built-in method vsplit of type object at 0x7fbe852b0540>

@implements(torch.Tensor.split)
@implements(torch.functional.split)
@implements(torch.split)
def split(tensor: StreamTensor, split_size_or_sections: Union[int, List[int]], dim: int = 0):
    
    state = tensor.stream_state
    tensor = tensor.tensor()
    
    if tensor.names[dim] in (LENGTH, BATCH):
        states = state.split(split_size_or_sections, tensor.names[dim])
        tensors = tensor.split(split_size_or_sections, dim=dim)
        assert len(tensors) == len(states)
        tensors = [StreamTensor(t, s) for t, s in zip(tensors, states)]
    else:
        tensors = tensor.split(split_size_or_sections, dim=dim)
        tensors = [StreamTensor(t, state) for t in tensors]
    
    return tensors

@implements(torch.unbind)
@implements(torch.Tensor.unbind)
def unbind(tensor: StreamTensor, dim=0):
    
    if tensor.names[dim] == LENGTH:
        # TODO: Implement this along with "split_length" for the StreamState.
        raise ValueError("Unbinding along the length dimension is not permitted - StreamTensors must have a length dimension.")

    state = tensor.stream_state
    tensor = tensor.tensor()
    
    if tensor.names[dim] == BATCH:
        states = state.unbind_batch()
        tensors = tensor.unbind(dim=dim)
        assert len(tensors) == len(states)
        tensors = [StreamTensor(t, s) for t, s in zip(tensors, states)]
    else:
        tensors = tensor.unbind(dim=dim)
        tensors = [StreamTensor(t, state) for t in tensors]
    
    return tensors

@implements(torch.chunk)
def chunk(tensor: StreamTensor, chunks: int, dim=0):
    
    raise NotImplementedError("Chunking is not yet supported.")

# TODO: Implement support for narrow - used in unpad_sequence:

@implements(torch.nn.functional.pad)
def pad(input: StreamTensor, pad: List[int], mode: str = 'constant', value: float = None):
    state = input.stream_state
    names = input.names
    input = input.tensor().rename(None)
    # TODO: Adapt stream_state
    output = torch.nn.functional.pad(input, pad, mode=mode, value=value)
    output = output.rename(*names)
    return StreamTensor(output, state)


@implements(torch.conv1d)
def conv1d(input: StreamTensor, *args, **kwargs):
    state = input.stream_state
    names = input.names
    # TODO: Adapt stream_state
    input = input.tensor().rename(None)
    output = torch.conv1d(input, *args, **kwargs)
    output = output.rename(*names)
    return StreamTensor(output, state)


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
# @implements(torch.sum)  # Fail if batch or length dim was removed? Maybe not.
# @implements(torch.mean)
# @implements(torch.std)
# @implements(torch.var)
# @implements(torch.median)
# @implements(torch.topk)

# indexing, slicing, joining, mutating methods
# @implements(torch.index)
# @implements(torch.gather)
# @implements(torch.scatter)
# @implements(torch.gather_index)

# moving dimensions
# @implements(torch.transpose)
# @implements(torch.permute)
