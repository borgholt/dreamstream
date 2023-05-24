import functools

from typing import Any, List, Tuple, Union

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
                raise NotImplementedError(
                    "Concatenating along the length dimension is only supported for non-padded tensors."
                )

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
# torch._VF.split
# torch._VF.split_with_sizes

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
        raise ValueError(
            "Unbinding along the length dimension is not permitted - StreamTensors must have a length dimension."
        )

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
def pad(input: StreamTensor, pad: List[int], mode: str = "constant", value: float = None):
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


def is_multidimensional_indexing(indices: Union[None, int, slice, List[Any], Tuple[Any, ...]]):
    """Return True if any of the indices are multidimensional, i.e. not an int, slice, or list/tuple of ints but 
    instead a list or tuple of those."""
    return isinstance(indices, (list, tuple)) and not all(isinstance(i, (int, bool)) for i in indices)


def any_index_is_multidimensional_tensor(indices: Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]]):
    """Return True if any of the indices are a multidimensional tensor."""
    if isinstance(indices, torch.Tensor) and indices.ndim > 1:
        return True
    if isinstance(indices, (list, tuple)):
        return any(any_index_is_multidimensional_tensor(i) for i in indices)


def determine_dims_affected_by_indexing(tensor: StreamTensor, indices: Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]], recursive_dim: int = None, return_only_new_names: bool = False) -> Tuple[List[int], Tuple[str, ...]]:
    """Return the dimensions of the tensor that are affected by indexing with `indices`.
    
    Also returns the names of the dimensions of the tensor resulting from the indexing. If the indexing operation
    removes a dimension, the name of that dimension is not included in the returned names. If the indexing operation
    adds a dimension, the name of that dimension is None. 
    
    If `recursive_dim` is given, only names of dimensions that are affected and remain after indexing are returned.

    - int: Selects a single index along the first dimension.
    - slice: Selects a range of indices along the first dimension.
    - List[int]: Coordinate indexing along a dimension. 
    - Tuple[int, ...]: Same as list[int]
    - IntTensor: Always indexes along the first dimension. If the IntTensor is ND for N>1, N - 1 new dimensions are
    -   inserted before the first dimension with 
    - BoolTensor: Selects all indices where the value is True. This flattens the indexed dimensions into a single 
    -   dimension with length equal to the total number of True values.
    - Tuple[Union[int, slice, Tensor, List[int], Tuple[int, ...]], ...]: A number of indexing operations on different
    -   dimensions.
    - List[Union[int, slice, Tensor, List[int], Tuple[int, ...]]]: Same as Tuple[Union[...], ...] but with a list.
    - Ellipsis: Selects all indices along the first dimension. This is equivalent to using :.
    - None: Same as Ellipsis.
    - slice(None): Same as Ellipsis.
    - slice(None, None): Same as Ellipsis.
    - slice(None, None, None): Same as Ellipsis.
    - slice(None, None, 2): Select every other element along the first dimension.
    - slice(None, None, -1): Reverses the first dimension.
    - slice(None, None, -2): Reverses the first dimension and removes every other element.
    """
    
    is_recursive = recursive_dim is not None
    recursive_dim = 0 if recursive_dim is None else recursive_dim
    names = tensor.names

    # If indices is None, slice(None) or Ellipsis, no dimensions are affected.
    if indices is None or indices == slice(None) or indices is Ellipsis:
        return [], [names[recursive_dim]] if is_recursive else names

    # If indices is an int or a slice, or a 1D IntTensor, or a 1D BoolTensor, only the first dimension is affected.
    if isinstance(indices, (int, slice)) or (isinstance(indices, Tensor) and indices.ndim == 1):
        if isinstance(indices, int):
            return [recursive_dim], [] if is_recursive else [n for i, n in enumerate(names) if i != recursive_dim]
        return [recursive_dim], [names[recursive_dim]] if is_recursive else names

    # If indices is a List[int] or a Tuple[int, ...], indexing is coordinate indexing along the first dimension.
    if isinstance(indices, (list, tuple)) and all(isinstance(i, (int, bool)) for i in indices):
        return [recursive_dim], [names[recursive_dim]] if is_recursive else names

    # If indices is a BoolTensor with N>1 dimensions, indexing starts from the first dimension and affects the next 
    # `indices.ndim` dimensions to the right. All affected dimensions are flattened into a single dimension with length
    # equal to the total number of True values in `indices`.
    if isinstance(indices, torch.Tensor) and indices.dtype == torch.bool and indices.ndim > 1:
        names = (None,) if is_recursive else names[:recursive_dim] + (None,) + names[recursive_dim + indices.ndim:]
        return list(range(recursive_dim, recursive_dim + indices.ndim)), names

    # If indices is an IntTensor with N>1 dimensions, N-1 new dimensions are inserted before the first dimension. But 
    # still, only the first dimension is affected.
    if isinstance(indices, torch.Tensor) and not torch.is_floating_point(indices) and indices.ndim > 1:
        names = (None,) * (indices.ndim - 1) + (names[recursive_dim],) if is_recursive else (None,) * (indices.ndim - 1) + names
        return [recursive_dim], names

    # If indices is a List[Union[int, slice, Tensor, List[int], Tuple[int, ...]]] or a Tuple[Union[int, slice, Tensor,
    # List[int], Tuple[int, ...]], ...], indexing is a number of indexing operations on different dimensions.
    if is_multidimensional_indexing(indices):
        dims_affected = []
        new_names = []
        for i, index in enumerate(indices):
            dims, names = determine_dims_affected_by_indexing(tensor, index, recursive_dim=i)
            dims_affected.extend(dims)
            new_names.extend(names)
            # print(names, new_names)

        return dims_affected, new_names

    msg = f"Indexing with {indices} is not supported. If you think this is a bug, please open an issue on GitHub."
    raise NotImplementedError(msg)
    

@implements(torch.Tensor.__getitem__)
def __getitem__(self: StreamTensor, indices: Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]]) -> StreamTensor:

    stream_state = self.stream_state
    tensor = self.tensor().rename(None)
    indexed_tensor = tensor[indices]

    dims_affected, new_names = determine_dims_affected_by_indexing(self, indices)

    if new_names:
        indexed_tensor = indexed_tensor.refine_names(*new_names)
    else:
        indexed_tensor = indexed_tensor.rename(*self.names)

    batch_dim = self.batch_dim
    length_dim = self.length_dim
    any_is_batch_dim = any([dim == batch_dim for dim in dims_affected])
    any_is_length_dim = any([dim == length_dim for dim in dims_affected])
    # import IPython
    # IPython.embed(using=False)

    if any_is_batch_dim or any_is_length_dim:
        if any_index_is_multidimensional_tensor(indices):
            msg = "Indexing with a >1D tensor that affects the batch or length dimensions is not supported."
            raise NotImplementedError(msg)

        # TODO (JDH): Handle simultaneous indexing along the batch and length dimensions.
        # e.g.
        # >>> stream_state[index]
        # where index is Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]] and indexes batch or length or both
        if any_is_length_dim:
            # Get the index that was used along the length dimension of the tensor.
            index = indices[length_dim] if is_multidimensional_indexing(indices) else indices
            stream_state = stream_state.index_length(index)

        if any_is_batch_dim:
            # Get the index that was used along the batch dimension of the tensor.
            index = indices[batch_dim] if is_multidimensional_indexing(indices) else indices
            stream_state = stream_state.index_batch(index)
    else:
        stream_state = stream_state.clone()

    stream_tensor = StreamTensor(indexed_tensor, stream_state)
    return stream_tensor


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
