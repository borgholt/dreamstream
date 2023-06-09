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
@implements(torch.concat)
@implements(torch.concatenate)
def cat(tensors: List[Union[StreamTensor, Tensor]], dim=0, *, out=None):
    """If dim is the batch dimension of any StreamTensor, assert all are StreamTensors and concatenate the stream
    states as well. Else, call torch.cat.
    """
    if len(tensors) == 1:
        return tensors[0]

    # Concatenation of at least one StreamTensor along the batch dimension.
    is_batch_dim = [t.is_batch_dim(dim) for t in tensors if isinstance(t, StreamTensor)]
    if any(is_batch_dim):
        require_all_stream_tensors(tensors, "Cannot concatenate StreamTensor and torch.Tensor along batch dimension.")
        tensors = [t.named_tensor() for t in tensors]
        tensor = torch.cat(tensors, dim=dim, out=out)
        meta = StreamMetadata.cat_batch([t.meta for t in tensors])  # TODO (JDH): Make this lazily evaluated.
        return StreamTensor(tensor, meta)

    # Concatenation of at least one StreamTensor along the length dimension.
    is_length_dim = [t.is_length_dim(dim) for t in tensors if isinstance(t, StreamTensor)]
    if any(is_length_dim):
        for t in tensors[:-1]:
            if isinstance(t, StreamTensor) and t.meta.lengths.min() < t.max_length():
                raise NotImplementedError("Only the last tensor can be padded when concatenating along length.")
        tensors = [t.named_tensor() if isinstance(t, StreamTensor) else t for t in tensors]
        tensor = torch.cat(tensors, dim=dim, out=out)
        meta = StreamMetadata.cat_length([t.meta for t in tensors if isinstance(t, StreamTensor)])
        meta.lengths += sum([t.size(dim) for t in tensors if not isinstance(t, StreamTensor)])
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


@implements(torch.nn.functional.pad)
def pad(input: StreamTensor, pad: List[int], mode: str = "constant", value: float = None):
    meta = input.meta
    names = input.names
    input = input.named_tensor().rename(None)
    # TODO: Adapt meta
    output = torch.nn.functional.pad(input, pad, mode=mode, value=value)
    output = output.rename(*names)
    return StreamTensor(output, meta)


# indexing, slicing, joining, mutating methods
@implements(torch.narrow)
@implements(torch.Tensor.narrow)
def narrow(input: StreamTensor, dim: int, start: int, length: int):
    tensor, meta = input.named_tensor(), input.meta
    out = tensor.narrow(dim, start, length)

    if tensor.names[dim] not in (LENGTH, BATCH):
        return StreamTensor(out, meta.clone())

    if tensor.names[dim] == BATCH:
        meta = meta[start : start + length]
    else:
        meta = meta[:, start : start + length]

    return StreamTensor(out, meta)


@implements(torch.gather)
@implements(torch.Tensor.gather)
def gather(input: StreamTensor, dim: int, index: torch.LongTensor, sparse_grad: bool = False, out: Optional[torch.Tensor] = None):
    tensor, meta, names = input.decouple()
    out = torch.gather(tensor,  dim, index, sparse_grad=sparse_grad, out=out)
    out = out.rename(*names)
    
    dim_is_batch_or_length = names[dim] in (LENGTH, BATCH)
    if dim_is_batch_or_length:
        # NOTE (JDH): Gathering along the batch dim can result mixing feature or time steps from different sequences.
        # NOTE (JDH): Gathering along the length dim can result in mixing feature or batch elements over time.
        raise IndexError("Gathering along the length or batch dimension is currently not supported for StreamTensors.")

    if BATCH in names:
        batch_dim = names.index(BATCH)
        index_batch_size = index.size(batch_dim)
        size_is_full_along_batch = index_batch_size == tensor.size(batch_dim)
    else:
        size_is_full_along_batch = True

    if LENGTH in names:
        length_dim = names.index(LENGTH)
        index_length_size = index.size(length_dim)
        size_is_full_along_length = index_length_size == tensor.size(length_dim)
    else:
        size_is_full_along_length = True

    if size_is_full_along_batch and size_is_full_along_length:
        return StreamTensor(out, meta.clone())
    
    # Gather removes elements beyond the size of index along the dimensions different from `dim`.
    batch_index = None if size_is_full_along_batch else slice(None, index_batch_size)
    length_index = None if size_is_full_along_length else slice(None, index_length_size)

    meta = meta[batch_index, length_index]

    return StreamTensor(out, meta)


# @implements(torch.scatter)

# @implements(torch.select)  # Equivalent to slicing with torch.Tensor.__getitem__
# @implements(torch.select_copy)

# @implements(torch.index_select)
# @implements(torch.masked_select)
# @implements(torch.take)
# @implements(torch.take_along_dim)
# @implements(torch.where)


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


def is_indexing_multidimensional(indices: Union[None, int, slice, List[Any], Tuple[Any, ...]]):
    """Return True if any of the indices are multidimensional, i.e. not an int, slice, or list/tuple of ints but
    instead a list or tuple of those."""
    return (
        indices is Ellipsis
        or isinstance(indices, (list, tuple))
        and not all(isinstance(i, (int, bool)) for i in indices)
    )


def get_locations_of_multidimensional_booltensors(indices: Union[List[Any], Tuple[Any, ...]]) -> List[int]:
    """Return the locations (linear index into `indices`) of any multidimensional booltensors"""
    indexes = []
    for i, index in enumerate(indices):
        if isinstance(index, torch.BoolTensor) and index.ndim > 1:
            indexes.append(i)
    return indexes


def replace_ellipsis(indices: Union[Tuple[Any, ...], List[Any]], ndim: int) -> List[Any]:
    """Replace Ellipsis in indices with the equivalent number of slices given the dimensionality of a torch.Tensor."""
    if Ellipsis not in indices:
        return indices

    # Always act on a list copy of the indices.
    if isinstance(indices, tuple):
        new_indices = list(indices)
    else:
        new_indices = deepcopy(indices)

    num_ellipsis = new_indices.count(Ellipsis)

    if ndim < len(new_indices) - num_ellipsis:
        raise IndexError(f"Too many indices for tensor of dimension {ndim}.")

    # If there are more than one Ellipsis, we infer the equivalent number of dimensions covered by each one.
    if num_ellipsis > 1:
        # Collapse neighboring Ellipsis into a single Ellipsis
        ellipsis_indices = [i for i, index in enumerate(new_indices) if index is Ellipsis]
        for i in reversed(range(num_ellipsis - 1)):
            if ellipsis_indices[i] + 1 == ellipsis_indices[i + 1]:
                new_indices[ellipsis_indices[i]] = Ellipsis
                del new_indices[ellipsis_indices[i] + 1]

        num_ellipsis = new_indices.count(Ellipsis)

        # At this point, we can only handle multiple remaining Ellipsis if they each correspond to one dimension.
        if num_ellipsis > 1:
            if len(new_indices) != ndim:
                if len(new_indices) > ndim:
                    raise IndexError(f"Too many indices for tensor of dimension {ndim}.")
                raise IndexError(f"Cannot resolve Ellipsis (got {num_ellipsis} Ellipsis for {ndim} dimensions)")

            # Replace each Ellipsis with a None slice
            for i in range(num_ellipsis):
                new_indices[ellipsis_indices[i]] = None

            return new_indices

    # Replace Ellipsis with as many Nones as needed to make the indices have the same length as the tensor
    ellipsis_index = new_indices.index(Ellipsis)
    num_missing_indices = ndim - len(new_indices) + 1
    new_indices = new_indices[:ellipsis_index] + [None] * num_missing_indices + new_indices[ellipsis_index + 1 :]

    return new_indices


def join_dim_names(*names: Union[Tuple[str, ...], List[str]]):
    """Join dimension names into a single string, separated by underscores."""
    return "_".join(names)


def determine_dims_affected_by_indexing(
    tensor: StreamTensor,
    indices: Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]],
    recursive_dim: int = None,
) -> Tuple[List[int], Tuple[str, ...]]:
    """Return the dimensions of the tensor that are affected by indexing with `indices`.

    Also returns the names of the new dimensions of the tensor that will result from the indexing.
    - If a dimension is removed, the name of that dimension is not included in the returned names.
    - If a dimension is added, the name of that dimension is None.
    - If two or more dimensions are merged into one, the new dimension is named by joining the original names.
    If `recursive_dim` is given, only names of dimensions that are affected and remain after indexing are returned.

    Finally, also returns an updated version of `indices` that has had any Ellipsis replaced with the appropriate
    number of Nones. This is needed for subsequent indexing into the StreamState.

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
        return indices, [], [names[first_dim]] if is_recursive else names, False

    if indices is Ellipsis:
        if is_recursive:
            raise RuntimeError("Ellipsis (...) should have been replaced with one or more `None` at this point...")
        return indices, [], names, False

    # If indices is an int or a slice, or a 1D IntTensor, or a 1D BoolTensor, only the first dimension is affected.
    if isinstance(indices, (int, slice)) or (isinstance(indices, torch.Tensor) and indices.ndim == 1):
        if isinstance(indices, int):
            return (
                indices,
                [first_dim],
                [] if is_recursive else [n for i, n in enumerate(names) if i != first_dim],
                False,
            )
        return indices, [first_dim], [names[first_dim]] if is_recursive else names, False

    # If indices is a List[int] or a Tuple[int, ...], indexing is coordinate indexing along the first dimension.
    if isinstance(indices, (list, tuple)) and all(isinstance(i, (int, bool)) for i in indices):
        return indices, [first_dim], [names[first_dim]] if is_recursive else names, False

    # If indices is a BoolTensor with N>1 dimensions, indexing starts from the first dimension and affects the next
    # `indices.ndim` dimensions to the right. All affected dimensions are flattened into a single dimension with size
    # equal to the total number of True values in `indices`.
    if isinstance(indices, torch.Tensor) and indices.dtype == torch.bool and indices.ndim > 1:
        new_name = join_dim_names(*names[first_dim : first_dim + indices.ndim])
        names = (new_name,) if is_recursive else (new_name,) + names[first_dim + indices.ndim :]
        return indices, list(range(first_dim, first_dim + indices.ndim)), names, False

    # If indices is an IntTensor with N>1 dimensions, N-1 new dimensions are inserted before the first dimension. But
    # still, only the first dimension is affected.
    if isinstance(indices, torch.Tensor) and not torch.is_floating_point(indices) and indices.ndim > 1:
        print(names[first_dim])
        new_none_dims = (None,) * (indices.ndim - 1)
        names = new_none_dims + (names[first_dim],) if is_recursive else new_none_dims + names
        return indices, [first_dim], names, False

    # If indices is a List[Union[int, slice, Tensor, List[int], Tuple[int, ...]]] or a Tuple[Union[int, slice, Tensor,
    # List[int], Tuple[int, ...]], ...], indexing is a number of indexing operations on different dimensions.
    if is_indexing_multidimensional(indices):
        # We need to replace Ellipsis with the equivalent number of single-dim slices because it can be used to index
        # multiple dimensions depending on the structure of the other indices.
        # indices = expand_indices(indices, len(names))
        indices = replace_ellipsis(indices, len(names))

        affected_dims = []
        new_names = []
        for i, index in enumerate(indices):
            _, dims, nms, _ = determine_dims_affected_by_indexing(tensor, index, recursive_dim=i)
            affected_dims.append(dims)
            new_names.extend(nms)

        # Some dimensions may not have been affected by any of the indices. In that case, we need to add their names
        # to the new names list. We can do this by adding names from the end of the original names list.
        num_missing = len(names) - len(indices)
        if isinstance(indices[-1], torch.BoolTensor):
            num_missing -= indices[-1].ndim - 1  # BoolTensors affect `ndim` dimensions.

        if num_missing > 0:
            new_names.extend(names[-num_missing:])

        return indices, affected_dims, new_names, True

    err_msg = f"Indexing with {indices} is not supported. If you think this is a bug, please open an issue on GitHub."
    raise NotImplementedError(err_msg)


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
    different lengths, sos and eos values that reflect whichever length indices were selected.

    The indexing operation will fail if
    - the length dimension is not indexed chronologically
    - the batch or length dimension is indexed with a multidimensional integer tensor


    # Cases for indexing:
    # - Any index is a multidimensional tensor
    #   - Tensor is bool
    #     - Resolve which dimensions are affected by each multidimensional tensor.
    #       - Tensor affects only batch and unimportant dimensions
    #       - Tensor affects only length and unimportant dimensions
    #       - Tensor affects both batch and length and potentially unimportant dimensions
    #   - Tensor is integer
    #     - Adds new dimensions left of the indexed dimension
    #     - If the indexed dimension is batch or length, this is not allowed.
    # X Single index on batch
    # X Single index on length
    # X Multidimensional indexing only affecting batch
    # X Multidimensional indexing only affecting length
    # X Multidimensional indexing affecting both batch and length
    """
    tensor, meta, names = self.decouple(copy_meta=False)  # TODO (JDH): A bit slow.

    expanded_indices, affected_dims, names, is_multidim_indexing = determine_dims_affected_by_indexing(self, indices)

    if is_multidim_indexing:
        dims_affected_flat = [dim for dims in affected_dims for dim in dims]
    else:
        dims_affected_flat = affected_dims
        affected_dims = [affected_dims]
        expanded_indices = [expanded_indices]

    indexed_tensor = tensor[indices]
    if names:
        indexed_tensor = indexed_tensor.refine_names(*names)
    else:
        indexed_tensor = indexed_tensor.rename(*self.names)

    batch_dim = self.batch_dim
    length_dim = self.length_dim
    is_batch_dim_affected = any([dim == batch_dim for dim in dims_affected_flat])
    is_length_dim_affected = any([dim == length_dim for dim in dims_affected_flat])

    if not (is_batch_dim_affected or is_length_dim_affected):
        # Indexing operation does not affect the batch or length dimensions, return the indexed tensor with same meta.
        return StreamTensor(indexed_tensor, meta.clone())

    # TODO (JDH): We must know if a single multidimensional tensor affected both batch and length dimensions.
    # TODO (JDH): We must know which dimension of an ndim tensor is the batch dimension and/or which is the length.
    # TODO (JDH): Deal with batch_dim to the right of length_dim. Swap them or tranpose a shared BoolTensor.

    # Placeholder variables for the indices that affect the batch and/or length dimensions.
    index = None
    batch_index = None
    length_index = None

    # Handle multidimensional boolean tensor indexing
    multidim_booltensor_locations = get_locations_of_multidimensional_booltensors(expanded_indices)
    if multidim_booltensor_locations:
        dims_and_indices = [
            (tensor_loc, affected_dims[tensor_loc], expanded_indices[tensor_loc])
            for tensor_loc in multidim_booltensor_locations
        ]  # Keep only multidimensional tensors that affect batch and/or length.
        for tensor_loc, dims, tensor_index in dims_and_indices:
            match (batch_dim in dims, length_dim in dims):
                case (True, True) if tensor_index.ndim == 2:
                    # No reduction needed, just use the tensor index.
                    index = tensor_index
                    break  # No need to continue the loop.
                case (True, True) if tensor_index.ndim > 2:
                    # Reduce with any along all dimensions except the batch and length dimensions.
                    all_but_batch_and_length_dim = [
                        dim - tensor_loc for dim in dims if dim not in (batch_dim, length_dim)
                    ]
                    index = tensor_index.sum(all_but_batch_and_length_dim) > 0
                    break  # No need to continue the loop.
                case (True, False):
                    # Reduce with any along all dimensions except the batch dimension.
                    all_but_batch_dim = [dim - tensor_loc for dim in dims if dim != batch_dim]
                    batch_index = tensor_index.sum(all_but_batch_dim) > 0
                case (False, True):
                    # Reduce with any along all dimensions except the length dimension.
                    all_but_length_dim = [dim - tensor_loc for dim in dims if dim != length_dim]
                    length_index = tensor_index.sum(all_but_length_dim) > 0

        if index is not None:
            return StreamTensor(indexed_tensor, meta[index])

    # Handle any other indexing operation
    if length_index is None and is_length_dim_affected:
        # Get the index that was used along the length dimension of the tensor.
        length_index = expanded_indices[length_dim] if is_multidim_indexing else expanded_indices[0]

    if batch_index is None and is_batch_dim_affected:
        # Get the index that was used along the batch dimension of the tensor.
        batch_index = expanded_indices[batch_dim] if is_multidim_indexing else expanded_indices[0]

    return StreamTensor(indexed_tensor, meta[batch_index, length_index])


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

# moving dimensions
# @implements(torch.transpose)
# @implements(torch.permute)
