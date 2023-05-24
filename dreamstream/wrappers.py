from copy import deepcopy

from typing import Optional, List, Union

from dreamstream.tensor import StreamTensor, StreamState, implements, require_all_stream_tensors
from dreamstream.utils.flags import BATCH, LENGTH

import torch
from torch import Tensor
import torch.nn.functional as F


    

@implements(torch.cat)
def cat(tensors: List[Union[StreamTensor, Tensor]], dim=0, *, out=None):
    """If dim is the batch dimension of any StreamTensor, assert all are StreamTensors and concatenate the stream 
    states as well. Else, call torch.cat.
    """
    
    # Mirror torch.cat's error messages.
    for t in tensors:
        if not (-t.ndim <= dim < t.ndim):
            raise IndexError(f"Dimension out of range (expected to be in range of [{-t.ndim}, {t.ndim-1}], but got {dim}.")
    
    if len(tensors) == 1:
        return tensors[0]
    
    # Concatenation of at least one StreamTensor along the batch dimension.
    is_batch_dim = [t._is_batch_dim(dim) for t in tensors if isinstance(t, StreamTensor)]
    if any(is_batch_dim):
        require_all_stream_tensors(tensors, "Cannot concatenate StreamTensor and torch.Tensor along batch dimension.")
        stream_state = StreamState.cat_batch([t.stream_state for t in tensors])  # TODO (JDH): Make this lazily evaluated.
        tensors = [t.named_tensor() for t in tensors]
        tensor = torch.cat(tensors, dim=dim, out=out)
        return StreamTensor(tensor, stream_state)

    # Concatenation of at least one StreamTensor along the length dimension.
    is_length_dim = [t.is_length_dim(dim) for t in tensors if isinstance(t, StreamTensor)]
    if any(is_length_dim):
        for t in tensors[:-1]:
            if isinstance(t, StreamTensor) and t.stream_state.lengths.min() < t.max_length():
                raise NotImplementedError("Only the last tensor can be padded when concatenating along the length dimension.")
        stream_state = StreamState.cat_length([t.stream_state for t in tensors if isinstance(t, StreamTensor)])
        stream_state.lengths += sum([t.size(dim) for t in tensors if not isinstance(t, StreamTensor)])
        tensors = [t.named_tensor() if isinstance(t, StreamTensor) else t for t in tensors]
        tensor = torch.cat(tensors, dim=dim, out=out)
        return StreamTensor(tensor, stream_state)

    # Concatenation along a dimension that is neither batch nor length.
    for t in tensors[:-1]:
        if not t.stream_state == tensors[0].stream_state:
            raise ValueError(f"It's ambiguos to concatenate tensors with different stream states along dim {dim}.")
    stream_state = deepcopy(tensors[0].stream_state)
    tensors = [t.named_tensor() if isinstance(t, StreamTensor) else t for t in tensors]
    tensor = torch.cat(tensors, dim=dim, out=out)
    return StreamTensor(tensor, stream_state)


@implements(torch.permute)
def permute(tensor: StreamTensor, dims: List[int]):
    names = [tensor.names[dim] for dim in dims]
    state = tensor.stream_state
    tensor = tensor.named_tensor().permute(*dims)
    return StreamTensor(tensor, state).rename(*names)

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
    tensor = tensor.named_tensor()
    
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
    tensor = tensor.named_tensor()
    
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
    input = input.named_tensor().rename(None)
    # TODO: Adapt stream_state
    output = torch.nn.functional.pad(input, pad, mode=mode, value=value)
    output = output.rename(*names)
    return StreamTensor(output, state)



def _compute_conv_output_lengths(input_lengths: Tensor, kernel_width: int, stride: int):
    return 


@implements(torch.conv1d)
def conv1d(input: StreamTensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
    
    # Convert padding to single integer value.
    if padding == "same":
        raise NotImplementedError("Same padding is not currently supported for StreamTensors.")
    elif padding == "valid":
        padding = 0
    elif isinstance(padding, tuple):
        padding = padding[0]
    
    assert padding >= 0, "Padding must be non-negative."
    
    # Compute kernel width and detach metadata and names. 
    kernel_width = weight.shape[2] + (weight.shape[2] - 1) * (dilation[0] - 1)
    state = input.stream_state
    names = input.names
    input = input.tensor()
    
    if padding > 0:
        
        # Adjust input lengths.
        if state.all_first_and_last:
            state.lengths += padding * 2
        elif state.any_first_or_last:
            state._first_lengths += padding
            state._last_lengths += padding
        
        # Apply padding.
        if not state.all_first_and_last:
            applied_padding = state.max_length - input.size(-1)        
            if state.all_first:
                assert applied_padding >= padding, "total padding should be greater than or equal to padding"
                input = F.pad(input, (padding, applied_padding - padding))
            elif applied_padding > 0:
                input = F.pad(input, (0, applied_padding))
                if state.any_first:
                    input[state.first] = torch.roll(input[state.first], shifts=padding, dims=-1)
            padding = 0     
    
    # Create buffer.
    output_lengths = ((state.lengths - kernel_width) // stride[0] + 1).clip(min=0)
    next_start = output_lengths * stride[0]
    buffer = {}
    if kernel_width > 1 and (not state.all_last):
        for i, (start, end, _id, is_last) in enumerate(zip(next_start, state.lengths, state.ids, state.is_last)):
            if not is_last:
                buffer[_id] = input[i, ..., start:end]
    
    # Convolve input and revert to StreamTensor.
    output = torch.conv1d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    output.rename_(*names)
    state.lengths = output_lengths
    #TODO: Consider whether to zero out the padding.
    return StreamTensor(output, state), buffer