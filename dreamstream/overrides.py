import functools
from copy import deepcopy
from typing import Any, Callable, Optional, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from dreamstream.tensor import StreamTensor, StreamMetadata
from dreamstream.func_coverage import OVERRIDDEN_FUNCTIONS
from dreamstream.utils.flags import BATCH, LENGTH
from dreamstream.warnings import fallback_operation_warning


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
    is_batch_dim = [t.is_batch_dim(dim) for t in tensors if isinstance(t, StreamTensor)]  # TODO (JDH): Speed this up
    if any(is_batch_dim):
        require_all_stream_tensors(tensors, "Cannot concatenate StreamTensor and torch.Tensor along batch dimension.")
        tensors = [t.named_tensor() for t in tensors]
        tensor = torch.cat(tensors, dim=dim, out=out)
        meta = StreamMetadata.cat_batch([t.meta for t in tensors])  # TODO (JDH): Make this lazily evaluated.
        return StreamTensor(tensor, meta)

    # Concatenation of at least one StreamTensor along the length dimension.
    is_length_dim = [t.is_length_dim(dim) for t in tensors if isinstance(t, StreamTensor)]  # TODO (JDH): Speed this up
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
    out = tensor.tensor().permute(*dims)
    meta = tensor.meta
    names = [tensor.names[dim] for dim in dims]
    return StreamTensor(out, meta).rename(*names)


@implements(torch.Tensor.permute)
def tensor_permute(tensor: StreamTensor, *dims: int):
    return permute(tensor, dims)


# Seemingly non-overrideable functions. Used in torch.Tensor.split:
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
def split(tensor: StreamTensor, split_size_or_sections: Union[int, List[int]], dim: int = 0) -> List[StreamTensor]:
    meta = tensor.meta
    tensor = tensor.named_tensor()

    if tensor.names[dim] in (LENGTH, BATCH):
        tensors = tensor.split(split_size_or_sections, dim=dim)
        metas = meta.split(split_size_or_sections, tensor.names[dim])
        tensors = [StreamTensor(t, s) for t, s in zip(tensors, metas)]
    else:
        tensors = tensor.split(split_size_or_sections, dim=dim)
        tensors = [StreamTensor(t, meta) for t in tensors]

    return tensors


# @implements(torch.tensor_split)
# @implements(torch.Tensor.tensor_split)
# def tensor_split(tensor: StreamTensor, indices_or_sections: Union[int, List[int]], dim: int = 0):
#     raise NotImplementedError("tensor_split is not implemented for StreamTensor.")


# @implements(torch.chunk)
# @implements(torch.Tensor.chunk)
# def chunk(tensor: StreamTensor, chunks: int, dim=0) -> List[StreamTensor]:
#     raise NotImplementedError("chunk is not implemented for StreamTensor.")


@implements(torch.unbind)
@implements(torch.Tensor.unbind)
def unbind(tensor: StreamTensor, dim=0) -> List[StreamTensor]:
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
        tensors = [StreamTensor(t, meta) for t, meta in zip(tensors, states)]
    else:
        tensors = tensor.unbind(dim=dim)
        tensors = [StreamTensor(t, meta) for t in tensors]

    return tensors


# @implements(torch.nn.functional.pad)
# def pad(input: StreamTensor, pad: List[int], mode: str = "constant", value: float = None):
#     raise NotImplementedError("pad is not currently supported for StreamTensors.")


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
        or (isinstance(indices, tuple))
        or (isinstance(indices, list) and not all(isinstance(i, (int, bool)) for i in indices))
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
    """Join dimension names into a single string, separated by underscores. Inverse of `split_dim_names`."""
    return "_".join(names)


def split_dim_names(name: str) -> Tuple[str, ...]:
    """Split a dimension name into a tuple of strings, separated by underscores. Inverse of `join_dim_names`."""
    return tuple(name.split("_"))


def determine_dims_affected_by_indexing(
    names: Tuple[str, ...],
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

    # If indices is None, slice(None), or Ellipsis no dimensions are affected.
    if indices is None or indices == slice(None):
        return indices, [], [names[first_dim]] if is_recursive else names, False

    if indices is Ellipsis:
        if is_recursive:
            raise RuntimeError("Ellipsis (...) should have been replaced with one or more `None` at this point...")
        return indices, [], names, False

    # If indices is an int or a slice, or a 1D IntTensor, or a 1D BoolTensor, only the first dimension is affected.
    if isinstance(indices, int):
        return indices, [first_dim], [] if is_recursive else [n for i, n in enumerate(names) if i != first_dim], False

    if isinstance(indices, slice) or (isinstance(indices, torch.Tensor) and indices.ndim == 1):
        return indices, [first_dim], [names[first_dim]] if is_recursive else names, False

    # If indices is a List[int] or a Tuple[int, ...] with `is_recursive == True`, indexing is coordinate indexing along
    # the first dimension. If indices is a Tuple[int, ...] and `is_recursive == False`, it is equivalent to
    # `tensor[*indices]` which is handled by `is_indexing_multidimensional` case below.)
    if (isinstance(indices, list) or (is_recursive and isinstance(indices, tuple))) and all(
        isinstance(i, (int, bool)) for i in indices
    ):
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
        new_none_dims = (None,) * (indices.ndim - 1)
        names = new_none_dims + (names[first_dim],) if is_recursive else new_none_dims + names
        return indices, [first_dim], names, False

    # If indices is a List[Union[int, slice, Tensor, List[int], Tuple[int, ...]]] or a Tuple[Union[int, slice, Tensor,
    # List[int], Tuple[int, ...]], ...], indexing is a number of indexing operations on different dimensions.
    # import IPython; IPython.embed(using=False)
    if is_indexing_multidimensional(indices):
        # We need to replace Ellipsis with the equivalent number of single-dim slices because it can be used to index
        # multiple dimensions depending on the structure of the other indices.
        # indices = expand_indices(indices, len(names))
        indices = replace_ellipsis(indices, len(names))

        affected_dims = []
        new_names = []
        for i, index in enumerate(indices):
            _, dims, nms, _ = determine_dims_affected_by_indexing(names, index, recursive_dim=i)
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


def any_along_dims(tensor: Tensor, dims: List[int]) -> Tensor:
    """Compute `torch.any` along the given dimensions."""
    if len(dims) == 0:
        return tensor

    if len(dims) == 1:
        return tensor.any(dims[0])

    return tensor.sum(dims) > 0
    # out = tensor.movedim(dims, tuple(-i for i in reversed(range(1, len(dims) + 1))))
    # out = out.flatten(start_dim=tensor.ndim - len(dims))
    # return out.any(-1)


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

    Cases for indexing:
    - Any index is a multidimensional tensor
      - Tensor is bool
        - Resolve which dimensions are affected by each multidimensional tensor.
          - Tensor affects only batch and unimportant dimensions
          - Tensor affects only length and unimportant dimensions
          - Tensor affects both batch and length and potentially unimportant dimensions
      - Tensor is integer
        - Adds new dimensions left of the indexed dimension
        - If the indexed dimension is batch or length, this is not allowed.
    - Single index on batch
    - Single index on length
    - Multidimensional indexing only affecting batch
    - Multidimensional indexing only affecting length
    - Multidimensional indexing affecting both batch and length
    """
    tensor, meta, names = self.decouple(copy_meta=False)

    indexed_tensor = tensor[indices]

    expanded_indices, affected_dims, names, is_multidim_indexing = determine_dims_affected_by_indexing(names, indices)

    if is_multidim_indexing:
        dims_affected_flat = [dim for dims in affected_dims for dim in dims]
    else:
        dims_affected_flat = affected_dims
        affected_dims = [affected_dims]
        expanded_indices = [expanded_indices]

    batch_dim = self.batch_dim
    length_dim = self.length_dim
    is_batch_dim_affected = any([dim == batch_dim for dim in dims_affected_flat])
    is_length_dim_affected = any([dim == length_dim for dim in dims_affected_flat])

    if not (is_batch_dim_affected or is_length_dim_affected):
        # Indexing operation does not affect the batch or length dimensions, return the indexed tensor with same meta.
        return StreamTensor(indexed_tensor.refine_names(*names), meta.clone())

    # Placeholder variables for the indices that affect the batch and/or length dimensions.
    batch_length_index = None
    batch_index = None
    length_index = None

    # Handle multidimensional boolean tensor indexing
    multidim_booltensor_locations = get_locations_of_multidimensional_booltensors(expanded_indices)
    if multidim_booltensor_locations:
        dims_and_indices = (
            (tensor_loc, affected_dims[tensor_loc], expanded_indices[tensor_loc])
            for tensor_loc in multidim_booltensor_locations
        )  # Keep only multidimensional tensors that affect batch and/or length.

        for tensor_loc, dims, tensor_index in dims_and_indices:
            match (batch_dim in dims, length_dim in dims):
                case (True, False):
                    # Reduce with any along all dimensions except the batch dimension.
                    non_batch_dims = [dim - tensor_loc for dim in dims if dim != batch_dim]
                    batch_index = any_along_dims(tensor_index, non_batch_dims)

                case (False, True):
                    # Reduce with any along all dimensions except the length dimension.
                    non_length_dims = [dim - tensor_loc for dim in dims if dim != length_dim]
                    length_index = any_along_dims(tensor_index, non_length_dims)

                # TODO (JDH): Simplify the batch+length case, no need for actually computing the index. Just a flag will do.
                case (True, True) if tensor_index.ndim > 2:
                    # Reduce with any along all dimensions except the batch and length dimensions.
                    non_batch_length_dims = [dim - tensor_loc for dim in dims if dim not in (batch_dim, length_dim)]
                    batch_length_index = any_along_dims(tensor_index, non_batch_length_dims)
                    break  # Found a 2D BoolTensor index on batch and length so remaining indices cannot affect them.

                case _:  # (True, True) if tensor_index.ndim == 2:
                    batch_length_index = tensor_index  # No reduction needed, just use the tensor batch_length_index.
                    break  # Found a 2D BoolTensor index on batch and length so remaining indices cannot affect them.

        if batch_length_index is not None:
            if batch_dim > length_dim:
                batch_length_index = batch_length_index.T  # Transpose if length comes before batch

            fallback_operation_warning(
                operation="multidimensional boolean tensor indexing",
                description="when the batch and length dimensions are affected by the indexing operation. "
                "This would create a mismatch with the StreamMetadata",
            )
            return indexed_tensor

    # Handle any other indexing operation
    if length_index is None and is_length_dim_affected:
        # Get the index that was used along the length dimension of the tensor.
        length_index = expanded_indices[length_dim] if is_multidim_indexing else expanded_indices[0]

    if batch_index is None and is_batch_dim_affected:
        # Get the index that was used along the batch dimension of the tensor.
        batch_index = expanded_indices[batch_dim] if is_multidim_indexing else expanded_indices[0]

    return StreamTensor(indexed_tensor.refine_names(*names), meta[batch_index, length_index])


# indexing, slicing, joining, mutating methods
@implements(torch.narrow)
@implements(torch.Tensor.narrow)
def narrow(input: StreamTensor, dim: int, start: int, length: int):
    tensor, meta = input.named_tensor(), input.meta
    out = tensor.narrow(dim, start, length)

    if tensor.names[dim] == BATCH:
        meta = meta[start : start + length]
    elif tensor.names[dim] == LENGTH:
        meta = meta[:, start : start + length]
    else:
        meta = meta.clone()

    return StreamTensor(out, meta)


@implements(torch.gather)
@implements(torch.Tensor.gather)
def gather(
    input: StreamTensor,
    dim: int,
    index: torch.LongTensor,
    sparse_grad: bool = False,
    out: Optional[torch.Tensor] = None,
) -> StreamTensor:
    return _gather_and_take_along_dim(input, dim, index=index, out=out, sparse_grad=sparse_grad, function=torch.gather)


@implements(torch.take_along_dim)
@implements(torch.Tensor.take_along_dim)
def take_along_dim(
    input: StreamTensor,
    indices: Tensor,
    dim: int,
    *,
    out: Optional[torch.Tensor] = None,
) -> StreamTensor:
    """torch.take_along_dim is a wrapper around torch.gather that has slightly different broadcast logic, see
    https://github.com/pytorch/pytorch/pull/52833."""
    return _gather_and_take_along_dim(input, dim, indices=indices, out=out, function=torch.take_along_dim)


def _gather_and_take_along_dim(
    input: StreamTensor,
    dim: int,
    out: Optional[torch.Tensor] = None,
    function: Callable = torch.gather,
    **kwargs,
):
    tensor, meta, names = input.decouple()
    out = function(tensor, dim=dim, out=out, **kwargs)
    index = kwargs.get("index", kwargs.get("indices"))

    dim_is_batch_or_length = names[dim] in (LENGTH, BATCH)
    if dim_is_batch_or_length:
        # NOTE (JDH): Gathering along the batch dim can result in mixing feature or time steps from different sequences.
        # NOTE (JDH): Gathering along the length dim can result in mixing feature or batch elements over time.
        # TODO (JDH): Fix this by printing the unsupported_operation_warning and returning a regular tensor.
        raise IndexError("Gathering along the length or batch dimension is currently not supported for StreamTensors.")

    if BATCH in names:
        batch_dim = names.index(BATCH)
        index_batch_size = index.size(batch_dim)
        if index_batch_size == tensor.size(batch_dim):
            batch_index = None
        else:
            # Gather removes elements beyond the size of index along the dimensions different from `dim`.
            batch_index = slice(None, index_batch_size)
    else:
        batch_index = None

    if LENGTH in names:
        length_dim = names.index(LENGTH)
        index_length_size = index.size(length_dim)
        if index_length_size == tensor.size(length_dim):
            length_index = None
        else:
            length_index = slice(None, index_length_size)
    else:
        length_index = None

    out = out.rename(*names)
    if batch_index is None and length_index is None:
        return StreamTensor(out, meta.clone())

    return StreamTensor(out, meta[batch_index, length_index])


@implements(torch.take)
@implements(torch.Tensor.take)
def take(input: StreamTensor, indices: Tensor, *, out: Optional[torch.Tensor] = None):
    tensor, meta, names = input.decouple()
    out = torch.take(tensor, indices, out=out)

    # Convert linear indices to subscripts (indices along each dimension).
    subs = np.unravel_index(indices, tensor.size())
    unique_subs = [list(dict.fromkeys(sub)) for sub in subs]  # remove duplicates, preserve order

    # Find the batch and length indices for meta.
    new_names = list(names)
    if BATCH in names:
        batch_dim = names.index(BATCH)
        batch_index = unique_subs[batch_dim]
        if len(batch_index) == 1:
            del new_names[batch_dim]
        if len(batch_index) == tensor.size(batch_dim):
            batch_index = None

    if LENGTH in names:
        length_dim = names.index(LENGTH)
        length_index = unique_subs[length_dim]
        if len(length_index) == 1:
            del new_names[length_dim]
        if len(length_index) == tensor.size(length_dim):
            length_index = None

    out = out.rename(join_dim_names(*new_names))

    if batch_index is None and length_index is None:
        return StreamTensor(out, meta.clone())

    return StreamTensor(out, meta[batch_index, length_index])


@implements(torch.select)
@implements(torch.Tensor.select)
@implements(torch.select_copy)
def select(input: StreamTensor, dim: int, index: Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]]):
    # TODO (JDH): Separate out torch.select and torch.Tensor.select which return views of the original data instead.
    dim = dim if dim >= 0 else input.ndim + dim
    getitem_index = (slice(None),) * dim + (index,)
    return input.__getitem__(getitem_index)


@implements(torch.masked_select)
def masked_select(input: StreamTensor, mask: Tensor, *, out: Optional[torch.Tensor] = None):
    tensor, meta, names = input.decouple()
    out = torch.masked_select(tensor, mask, out=out)
    fallback_operation_warning(
        operation="masked_select",
        description="since the StreamTensor becomes 1D which invalidates the link with the StreamMetadata.",
    )
    return out


@implements(torch.flatten)
@implements(torch.Tensor.flatten)
def flatten(input: StreamTensor, start_dim: int = 0, end_dim: int = -1) -> StreamTensor:
    tensor, meta, names = input.decouple()
    out = torch.flatten(tensor, start_dim=start_dim, end_dim=end_dim)

    affected_dims = names[start_dim : end_dim + 1]
    if BATCH in affected_dims or LENGTH in affected_dims:
        fallback_operation_warning("flatten", "when applied on the batch and/or length dimensions")
        return out

    names = names[:start_dim] + (join_dim_names(*affected_dims),) + names[end_dim + 1 :]
    out.rename_(*names)
    return StreamTensor(out, meta.clone())


@implements(torch.squeeze)
@implements(torch.Tensor.squeeze)
def squeeze(input: StreamTensor, dim: Optional[int] = None) -> StreamTensor:
    tensor, meta, names = input.decouple()
    size = tensor.size()

    # dim=None raises: `RuntimeError: Please look up dimensions by name, got: name = None.`, so we have to do this.
    if dim is None:
        tensor = torch.squeeze(tensor)
    else:
        tensor = torch.squeeze(tensor, dim=dim)

    if dim is None:
        # Remove all singleton dimensions.
        remove = set(names[i] for i, size in enumerate(size) if size == 1)
        names = [n for n in names if n not in remove]
    elif size[dim] == 1:
        # Remove the specified singleton dimension.
        names = list(names)
        del names[dim]

    tensor.rename_(*names)
    return StreamTensor(tensor, meta.clone())


@implements(torch.unsqueeze)
@implements(torch.Tensor.unsqueeze)
def unqsqueeze(input: StreamTensor, dim: int) -> StreamTensor:
    tensor, meta, names = input.decouple()
    tensor = torch.unsqueeze(tensor, dim=dim)
    names = list(names)
    names.insert(dim, None)
    tensor.rename_(*names)
    return StreamTensor(tensor, meta.clone())


# X @implements(torch.gather)
# X @implements(torch.narrow)

# @implements(torch.scatter)
# @implements(torch.diagonal_scatter)
# @implements(torch.select_scatter)
# @implements(torch.slice_scatter)

# @implements(torch.scatter_add)
# @implements(torch.scatter_add_)
# @implements(torch.scatter_reduce)
# @implements(torch.scatter_reduce_)

# X @implements(torch.select)  # Equivalent to slicing with torch.Tensor.__getitem__
# X @implements(torch.select_copy)

# @implements(torch.index_select)
# X @implements(torch.masked_select)
# X @implements(torch.take)
# X @implements(torch.take_along_dim)
# X @implements(torch.where)

# @implements(torch.stack)  # Create a new dim and name it
# @implements(torch.vstack)
# @implements(torch.hstack)

# X @implements(torch.split)
# @implements(torch.chunk)
# @implements(torch.tensor_split)

# X @implements(torch.flatten)

# X @implements(torch.squeeze)
# X @implements(torch.unsqueeze)  # Give new dim default name

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
