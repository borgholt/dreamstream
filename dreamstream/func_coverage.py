import torch


FLAT_OVERRIDABLE_FUNCTIONS = {f for k, fs in torch.overrides.get_overridable_functions().items() for f in fs}

# Functions that must be overridden to handle StreamMetadata and limit use cases.
OVERRIDDEN_FUNCTIONS = dict()

# Functions that work for StreamTensors using the super().__torch_function__.
VALID_FUNCTIONS = {
    torch.Tensor.__repr__,
    torch.Tensor.__str__,
}

# Functions that must be wrapped to avoid returning a StreamTensor.
DECOUPLE_FUNCTIONS = {
    torch.argmax,
    torch.argmin,
    torch.argsort,
}

# Functions that must be wrapped to avoid failures related to named tensors and to maintain StreamMetadata.
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
}

# The full set of functions that are explicitly supported for StreamTensors.
SUPPORTED_FUNCTIONS = VALID_FUNCTIONS | DECOUPLE_FUNCTIONS | RECOUPLE_FUNCTIONS | OVERRIDDEN_FUNCTIONS.keys()

# The set of functions that are overrideable but not explicitly supported. These functions may still work correctly.
UNSUPPORTED_FUNCTIONS = {f for f in FLAT_OVERRIDABLE_FUNCTIONS if f not in SUPPORTED_FUNCTIONS}
