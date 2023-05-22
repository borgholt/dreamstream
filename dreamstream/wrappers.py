from typing import List, Union

from dreamstream.tensor import StreamTensor, StreamState, implements, require_all_stream_tensors
from dreamstream.utils.flags import BATCH, LENGTH

import torch
from torch import Tensor


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
        
        raise NotImplementedError("Concatenating along the length dimension is not yet supported.")
    
    return torch.cat(tensors, dim=dim, out=out)


@implements(torch.permute)
def permute(tensor: StreamTensor, dims: Union[List[int], List[str]]):
    dims = [tensor.names[dim] for dim in dims]
    return tensor.align_to(*dims)

@implements(torch.Tensor.permute)
def tensor_permute(tensor: StreamTensor, *dims: Union[int, str]):
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
    
    if tensor.names[dim] == LENGTH:
        # TODO: Implement this along with "split_length" for the StreamState.
        raise ValueError("Splitting along the length dimension is not currently supported.")

    state = tensor.stream_state
    tensor = tensor.tensor()
    
    if tensor.names[dim] == BATCH:
        states = state.split_batch(split_size_or_sections)
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

# TODO: Implement support for narrow - used in unpad_sequence:
