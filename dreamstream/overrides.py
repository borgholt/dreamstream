import functools
from copy import deepcopy
from typing import Any, Optional, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from dreamstream.tensor import StreamTensor, StreamMetadata
from dreamstream.func_coverage import OVERRIDDEN_FUNCTIONS
from dreamstream.utils.flags import BATCH, LENGTH


# Wrap whatever functools.update_wrapper usually wraps, except __doc__
WRAPPER_ASSIGNMENTS = tuple(set(functools.WRAPPER_ASSIGNMENTS) - {"__doc__"})
TORCH_DOC_LINEWIDTH = 80


def all_stream_tensors(tensors: List[Union[StreamTensor, Tensor]]) -> bool:
    """Return True if all tensors are StreamTensors."""
    return all(isinstance(t, StreamTensor) for t in tensors)


def require_all_stream_tensors(tensors: List[Union[StreamTensor, Tensor]], message: str = None):
    """Raise an error if any of the tensors are not StreamTensors."""
    if not all_stream_tensors(tensors):
        message = message or "All tensors must be StreamTensors."
        raise ValueError(message)


def augment_documentation(torchstream_doc: str, torch_doc: str) -> str:
    """Augment the documentation of a function to include information about StreamTensors."""
    return f"""{torchstream_doc}\n\n{"=" * TORCH_DOC_LINEWIDTH}\n\n{torch_doc}"""


def implements(torch_function):
    """Register a torch function override for StreamTensor."""

    def decorator(func):
        functools.update_wrapper(func, torch_function, assigned=WRAPPER_ASSIGNMENTS)
        func.__doc__ = augment_documentation(func.__doc__, torch_function.__doc__)
        OVERRIDDEN_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements(torch.cat)
def cat(tensors: List[Union[StreamTensor, Tensor]], dim=0, *, out=None):
    """If dim is the batch dimension of any StreamTensor, assert all are StreamTensors and concatenate the stream
    states as well. Else, call torch.cat.
    """

    # Mirror torch.cat's error messages.
    for t in tensors:
        if not (-t.ndim <= dim < t.ndim):
            raise IndexError(f"Dimension out of range (expected to be in [{-t.ndim}, {t.ndim-1}], but got {dim}.")

    if len(tensors) == 1:
        return tensors[0]

    # Concatenation of at least one StreamTensor along the batch dimension.
    is_batch_dim = [t.is_batch_dim(dim) for t in tensors if isinstance(t, StreamTensor)]
    if any(is_batch_dim):
        require_all_stream_tensors(tensors, "Cannot concatenate StreamTensor and torch.Tensor along batch dimension.")
        meta = StreamMetadata.cat_batch([t.meta for t in tensors])  # TODO (JDH): Make this lazily evaluated.
        tensors = [t.named_tensor() for t in tensors]
        tensor = torch.cat(tensors, dim=dim, out=out)
        return StreamTensor(tensor, meta)

    # Concatenation of at least one StreamTensor along the length dimension.
    is_length_dim = [t.is_length_dim(dim) for t in tensors if isinstance(t, StreamTensor)]
    if any(is_length_dim):
        for t in tensors[:-1]:
            if isinstance(t, StreamTensor) and t.meta.lengths.min() < t.max_length():
                raise NotImplementedError("Only the last tensor can be padded when concatenating along length.")
        meta = StreamMetadata.cat_length([t.meta for t in tensors if isinstance(t, StreamTensor)])
        meta.lengths += sum([t.size(dim) for t in tensors if not isinstance(t, StreamTensor)])
        tensors = [t.named_tensor() if isinstance(t, StreamTensor) else t for t in tensors]
        tensor = torch.cat(tensors, dim=dim, out=out)
        return StreamTensor(tensor, meta)

    # Concatenation along a dimension that is neither batch nor length.
    for t in tensors[:-1]:
        if not t.meta == tensors[0].meta:
            raise ValueError(f"It's ambiguos to concatenate tensors with different stream states along dim {dim}.")
    meta = deepcopy(tensors[0].meta)
    tensors = [t.named_tensor() if isinstance(t, StreamTensor) else t for t in tensors]
    tensor = torch.cat(tensors, dim=dim, out=out)
    return StreamTensor(tensor, meta)


@implements(torch.permute)
def permute(tensor: StreamTensor, dims: List[int]):
    names = [tensor.names[dim] for dim in dims]
    meta = tensor.meta
    tensor = tensor.tensor().permute(*dims)
    return StreamTensor(tensor, meta).rename(*names)


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
    meta = tensor.meta
    tensor = tensor.named_tensor()

    if tensor.names[dim] in (LENGTH, BATCH):
        states = meta.split(split_size_or_sections, tensor.names[dim])
        tensors = tensor.split(split_size_or_sections, dim=dim)
        assert len(tensors) == len(states)
        tensors = [StreamTensor(t, s) for t, s in zip(tensors, states)]
    else:
        tensors = tensor.split(split_size_or_sections, dim=dim)
        tensors = [StreamTensor(t, meta) for t in tensors]

    return tensors


@implements(torch.unbind)
@implements(torch.Tensor.unbind)
def unbind(tensor: StreamTensor, dim=0):
    if tensor.names[dim] == LENGTH:
        # TODO: Implement this along with "split_length" for the StreamMetadata.
        raise ValueError(
            "Unbinding along the length dimension is not permitted - StreamTensors must have a length dimension."
        )

    meta = tensor.meta
    tensor = tensor.named_tensor()

    if tensor.names[dim] == BATCH:
        states = meta.unbind_batch()
        tensors = tensor.unbind(dim=dim)
        assert len(tensors) == len(states)
        tensors = [StreamTensor(t, s) for t, s in zip(tensors, states)]
    else:
        tensors = tensor.unbind(dim=dim)
        tensors = [StreamTensor(t, meta) for t in tensors]

    return tensors


@implements(torch.chunk)
def chunk(tensor: StreamTensor, chunks: int, dim=0):
    raise NotImplementedError("Chunking is not yet supported.")


# TODO: Implement support for narrow - used in unpad_sequence:


@implements(torch.nn.functional.pad)
def pad(input: StreamTensor, pad: List[int], mode: str = "constant", value: float = None):
    meta = input.meta
    names = input.names
    input = input.named_tensor().rename(None)
    # TODO: Adapt meta
    output = torch.nn.functional.pad(input, pad, mode=mode, value=value)
    output = output.rename(*names)
    return StreamTensor(output, meta)


def _compute_conv_output_lengths(input_lengths: Tensor, kernel_width: int, stride: int):
    return


@implements(torch.conv1d)
def conv1d(
    input: StreamTensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
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
    input, meta, names = input.decouple()

    if padding > 0:
        # Adjust input lengths.
        if meta.all_starting_and_ending:
            meta.lengths += padding * 2
        elif meta.any_starting_or_ending:
            meta.starting_lengths += padding
            meta.ending_lengths += padding

        # Apply padding.
        if not meta.all_starting_and_ending:
            applied_padding = meta.max_length - input.size(-1)
            if applied_padding > 0:
                if meta.all_starting:
                    assert applied_padding >= padding, "total padding should be greater than or equal to padding"
                    input = F.pad(input, (padding, applied_padding - padding))
                else:
                    input = F.pad(input, (0, applied_padding))
            if meta.any_starting and not meta.all_starting:
                input[meta.sos] = torch.roll(input[meta.sos], shifts=padding, dims=-1)
            padding = 0

    # Create buffer.
    output_lengths = ((meta.lengths - kernel_width) // stride[0] + 1).clip(min=0)
    next_start = output_lengths * stride[0]
    buffer = {}
    if kernel_width > 1 and (not meta.all_ending):
        for i, (start, end, _id, eos) in enumerate(zip(next_start, meta.lengths, meta.ids, meta.eos)):
            if not eos:
                buffer[_id] = input[i, ..., start:end]

    # Convolve input and revert to StreamTensor.
    output = torch.conv1d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    output.rename_(*names)
    meta.lengths = output_lengths
    # TODO: Consider whether to zero out the padding.
    return StreamTensor(output, meta), buffer


def is_multidimensional_indexing(indices: Union[None, int, slice, List[Any], Tuple[Any, ...]]):
    """Return True if any of the indices are multidimensional, i.e. not an int, slice, or list/tuple of ints but
    instead a list or tuple of those."""
    return (
        indices is Ellipsis
        or isinstance(indices, (list, tuple))
        and not all(isinstance(i, (int, bool)) for i in indices)
    )


def any_index_is_multidimensional_tensor(indices: Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]]):
    """Return True if any of the indices are a multidimensional tensor."""
    if isinstance(indices, torch.Tensor) and indices.ndim > 1:
        return True

    if isinstance(indices, (list, tuple)):
        return any(any_index_is_multidimensional_tensor(i) for i in indices)

    return False


def replace_ellipsis(indices: Union[Tuple[Any, ...], List[Any]], ndim: int) -> Union[List[Any]]:
    if Ellipsis not in indices:
        return indices

    new_indices = deepcopy(indices)
    if isinstance(new_indices, tuple):
        new_indices = list(new_indices)

    ellipsis_count = new_indices.count(Ellipsis)

    if ndim < len(new_indices) - ellipsis_count:
        raise IndexError(f"Too many indices for tensor of dimension {ndim}.")

    # If there are more than one Ellipsis, we infer the equivalent number of dimensions covered by each one.
    if ellipsis_count > 1:
        # Collapse neighboring Ellipsis into a single Ellipsis
        ellipsis_indices = [i for i, index in enumerate(new_indices) if index is Ellipsis]
        for i in reversed(range(ellipsis_count - 1)):
            if ellipsis_indices[i] + 1 == ellipsis_indices[i + 1]:
                new_indices[ellipsis_indices[i]] = Ellipsis
                del new_indices[ellipsis_indices[i] + 1]

        ellipsis_count = new_indices.count(Ellipsis)

        # We can only handle multiple remaining Ellipsis at this point, if they correspond to one dimension each.
        if ellipsis_count > 1:
            if len(new_indices) != ndim:
                if len(new_indices) > ndim:
                    raise IndexError(f"Too many indices for tensor of dimension {ndim}.")
                raise IndexError(f"Cannot resolve Ellipsis (got {ellipsis_count} Ellipsis for {ndim} dimensions)")

            # Replace each Ellipsis with a None slice
            for i in range(ellipsis_count):
                new_indices[ellipsis_indices[i]] = None

            return new_indices

    # Replace Ellipsis with as many Nones as needed to make the indices have the same length as the tensor
    ellipsis_index = new_indices.index(Ellipsis)
    num_missing_indices = ndim - len(new_indices) + 1
    new_indices = new_indices[:ellipsis_index] + [None] * num_missing_indices + new_indices[ellipsis_index + 1:]

    return new_indices


def determine_dims_affected_by_indexing(
    tensor: StreamTensor,
    indices: Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]],
    recursive_dim: int = None,
) -> Tuple[List[int], Tuple[str, ...]]:
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
    first_dim = 0 if recursive_dim is None else recursive_dim
    names = tensor.names

    # If indices is None, slice(None), or Ellipsis no dimensions are affected.
    if indices is None or indices == slice(None):
        return indices, [], [names[first_dim]] if is_recursive else names

    if indices is Ellipsis:
        if is_recursive:
            raise RuntimeError("Ellipsis (...) should have been replaced with one or more `None` at this point...")
        return indices, [], names

    # If indices is an int or a slice, or a 1D IntTensor, or a 1D BoolTensor, only the first dimension is affected.
    if isinstance(indices, (int, slice)) or (isinstance(indices, torch.Tensor) and indices.ndim == 1):
        if isinstance(indices, int):
            return indices, [first_dim], [] if is_recursive else [n for i, n in enumerate(names) if i != first_dim]
        return indices, [first_dim], [names[first_dim]] if is_recursive else names

    # If indices is a List[int] or a Tuple[int, ...], indexing is coordinate indexing along the first dimension.
    if isinstance(indices, (list, tuple)) and all(isinstance(i, (int, bool)) for i in indices):
        return indices, [first_dim], [names[first_dim]] if is_recursive else names

    # If indices is a BoolTensor with N>1 dimensions, indexing starts from the first dimension and affects the next
    # `indices.ndim` dimensions to the right. All affected dimensions are flattened into a single dimension with length
    # equal to the total number of True values in `indices`.
    if isinstance(indices, torch.Tensor) and indices.dtype == torch.bool and indices.ndim > 1:
        names = (None,) if is_recursive else names[:first_dim] + (None,) + names[first_dim + indices.ndim :]
        return indices, list(range(first_dim, first_dim + indices.ndim)), names

    # If indices is an IntTensor with N>1 dimensions, N-1 new dimensions are inserted before the first dimension. But
    # still, only the first dimension is affected.
    if isinstance(indices, torch.Tensor) and not torch.is_floating_point(indices) and indices.ndim > 1:
        new_none_dims = (None,) * (indices.ndim - 1)
        names = new_none_dims + (names[first_dim],) if is_recursive else new_none_dims + names
        return indices, [first_dim], names

    # If indices is a List[Union[int, slice, Tensor, List[int], Tuple[int, ...]]] or a Tuple[Union[int, slice, Tensor,
    # List[int], Tuple[int, ...]], ...], indexing is a number of indexing operations on different dimensions.
    if is_multidimensional_indexing(indices):
        # We need to replace Ellipsis with the equivalent number of single-dim slices because it can be used to index
        # multiple dimensions depending on the structure of the other indices.
        indices = replace_ellipsis(indices, len(names))

        dims_affected = []
        new_names = []
        for i, index in enumerate(indices):
            _, dims, names = determine_dims_affected_by_indexing(tensor, index, recursive_dim=i)
            dims_affected.extend(dims)
            new_names.extend(names)

        return indices, dims_affected, new_names

    msg = f"Indexing with {indices} is not supported. If you think this is a bug, please open an issue on GitHub."
    raise NotImplementedError(msg)


@implements(torch.Tensor.__getitem__)
def __getitem__(
    self: StreamTensor, indices: Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]]
) -> StreamTensor:
    """Index a StreamTensor along any of its dimensions.

    If the indexing operation does not affect the batch or length dimensions it is done as usual and the resulting
    tensor gets the same names and metadata as the original tensor.

    If the indexing operation affects the batch dimension, the resulting tensor will have metadata only for the
    files that remain after indexing.

    If the indexing operation affects the length dimension, the resulting tensor will have metadata with potentially
    different lengths, sos and eos values that reflect whichever length indices were selected. The indexing operation
    will fail if the length dimension is not indexed chronologically or if the length dimension is indexed with a
    multidimensional tensor.
    """

    tensor, meta, names = self.decouple(copy_meta=False)

    indexed_tensor = tensor[indices]

    new_indices, dims_affected, new_names = determine_dims_affected_by_indexing(self, indices)

    if new_names:
        indexed_tensor = indexed_tensor.refine_names(*new_names)
    else:
        indexed_tensor = indexed_tensor.rename(*self.names)

    batch_dim = self.batch_dim
    length_dim = self.length_dim
    any_is_batch_dim = any([dim == batch_dim for dim in dims_affected])
    any_is_length_dim = any([dim == length_dim for dim in dims_affected])

    if not (any_is_batch_dim or any_is_length_dim):
        # Indexing operation does not affect the batch or length dimensions, return the indexed tensor with same meta.
        return StreamTensor(indexed_tensor, meta.clone())

    if any_index_is_multidimensional_tensor(new_indices):
        import IPython
        IPython.embed(using=False)
        msg = "Indexing with a multidimensional tensor that affects the batch or length dimensions is not supported."
        raise NotImplementedError(msg)

    if any_is_length_dim:
        # Get the index that was used along the length dimension of the tensor.
        length_index = new_indices[length_dim] if is_multidimensional_indexing(new_indices) else new_indices
    else:
        length_index = None

    if any_is_batch_dim:
        # Get the index that was used along the batch dimension of the tensor.
        batch_index = new_indices[batch_dim] if is_multidimensional_indexing(new_indices) else new_indices
    else:
        batch_index = None

    meta = meta[batch_index, length_index]

    stream_tensor = StreamTensor(indexed_tensor, meta)
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
# @implements(torch.max)
# @implements(torch.min)
# @implements(torch.sum)
# @implements(torch.mean)
# @implements(torch.median)
# @implements(torch.mode)
# @implements(torch.std)
# @implements(torch.var)
# @implements(torch.prod)
# @implements(torch.norm)

# comparison methods
# @implements(torch.sort)
# @implements(torch.topk)
# @implements(torch.unique)

# indexing, slicing, joining, mutating methods
# @implements(torch.index)
# @implements(torch.gather)
# @implements(torch.scatter)
# @implements(torch.gather_index)

# moving dimensions
# @implements(torch.transpose)
# @implements(torch.permute)
