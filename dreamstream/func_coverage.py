import torch

from dreamstream.utils.listloaders import (
    load_default_valid_pointwise_ops,
    load_valid_pointwise_ops,
    load_recouple_pointwise_ops,
    load_inplace_recouple_pointwise_ops,
)


FLAT_OVERRIDABLE_FUNCTIONS = {f for k, fs in torch.overrides.get_overridable_functions().items() for f in fs}

# Functions that must be overridden to handle StreamMetadata and limit use cases.
CUSTOMIZED_FUNCTIONS = dict()

GET_METHODS = {f for f in FLAT_OVERRIDABLE_FUNCTIONS if f.__name__ == "__get__"}

DEFAULT_VALID_POITWISE_OPS = load_default_valid_pointwise_ops()
VALID_POITWISE_OPS = load_valid_pointwise_ops()
RECOUPLE_POITWISE_OPS = load_recouple_pointwise_ops()
INPLACE_RECOUPLE_POITWISE_OPS = load_inplace_recouple_pointwise_ops()

# Functions that work for StreamTensors using the super().__torch_function__.
VALID_FUNCTIONS = {
    torch.Tensor.__repr__,
    torch.Tensor.__str__,
    torch.Tensor.__dir__,
    torch.Tensor.size,
    torch.equal,
    torch.Tensor.rename_,
    torch.Tensor.zero_,
    torch.Tensor.fill_,
    torch.Tensor.bernoulli_,
    torch.Tensor.cauchy_,
    torch.Tensor.exponential_,
    torch.Tensor.geometric_,
    torch.Tensor.log_normal_,
    torch.Tensor.normal_,
    torch.Tensor.random_,
    torch.Tensor.uniform_,
    # torch.is_tensor and torch.is_storage are not overridable
    torch.is_complex,
    torch.Tensor.is_complex,
    torch.is_conj,
    torch.Tensor.is_conj,
    torch.is_floating_point,
    torch.Tensor.is_floating_point,
    torch.is_nonzero,
    torch.Tensor.is_nonzero,
    torch.numel,
    torch.Tensor.numel,
    torch.Tensor.dim,
    torch.Tensor.imag,
    torch.Tensor.real,
}
VALID_FUNCTIONS.update(GET_METHODS)
VALID_FUNCTIONS.update(VALID_POITWISE_OPS)

# Functions that work for StreamTensors using the super().__torch_function__, but .meta is not preserved.
DEFAULT_VALID_FUNCTIONS = {
    torch.Tensor.align_to,
    torch.transpose,
    torch.Tensor.transpose,
    torch.Tensor.rename,
    torch.clone,  # meta is not preserved
    torch.Tensor.clone,  # meta is not preserved
    torch.rand_like,
    torch.randn_like,
    torch.zeros_like,
    torch.ones_like,
    torch.empty_like,
    torch.full_like,
    torch.Tensor.__abs__,
    torch.Tensor.__neg__,
    torch.Tensor.__add__,
}
DEFAULT_VALID_FUNCTIONS.update(DEFAULT_VALID_POITWISE_OPS)

# Functions that must be wrapped to avoid returning a StreamTensor (and may not).
DECOUPLE_FUNCTIONS = {
    torch.argmax,
    torch.argmin,
    torch.argsort,
    torch.allclose,
    torch.Tensor.allclose,
    torch.quantize_per_channel,  # output "type" is not supported with names
    torch.fake_quantize_per_channel_affine,  # output "type" is not supported with names
    torch.fake_quantize_per_tensor_affine,  # output "type" is not supported with names
    torch.gradient,  # TODO: double check if decouple is the right category for torch.gradient
}

# Functions that must be wrapped to avoid failures related to named tensors (and to maintain StreamMetadata).
RECOUPLE_FUNCTIONS = {
    torch.adjoint,
    torch.sigmoid,
    torch.Tensor.sigmoid,
    torch.transpose,
    torch.where,
    torch.Tensor.where,
    torch.scatter,
    torch.Tensor.scatter,
    torch.Tensor.scatter_,
    torch.diagonal_scatter,
    torch.Tensor.diagonal_scatter,
    torch.select_scatter,
    torch.Tensor.select_scatter,
    torch.slice_scatter,
    torch.Tensor.slice_scatter,
    torch.isclose,
    torch.Tensor.isclose,
    torch.randint_like,
    torch.heaviside,
    torch.Tensor.heaviside,
    torch.dequantize,
    torch.Tensor.dequantize,
    torch.polar,
    torch.complex,
    torch.real,
    torch.imag,
}
RECOUPLE_FUNCTIONS.update(RECOUPLE_POITWISE_OPS)

INPLACE_RECOUPLE_FUNCTIONS = {
    torch.Tensor.heaviside_,
}
INPLACE_RECOUPLE_FUNCTIONS.update(INPLACE_RECOUPLE_POITWISE_OPS)

# The full set of functions that are explicitly supported for StreamTensors.
SUPPORTED_FUNCTIONS = (
    VALID_FUNCTIONS
    | DEFAULT_VALID_FUNCTIONS
    | DECOUPLE_FUNCTIONS
    | RECOUPLE_FUNCTIONS
    | INPLACE_RECOUPLE_FUNCTIONS
    | CUSTOMIZED_FUNCTIONS.keys()
)

# The set of functions that are overrideable but not explicitly supported. These functions may still work correctly.
UNSUPPORTED_FUNCTIONS = {f for f in FLAT_OVERRIDABLE_FUNCTIONS if f not in SUPPORTED_FUNCTIONS}
SUPPORTED_NON_OVERRIDEABLE_FUNCTIONS = {f for f in SUPPORTED_FUNCTIONS if f not in FLAT_OVERRIDABLE_FUNCTIONS}
