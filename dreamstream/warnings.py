import warnings


class TorchStreamWarning(RuntimeWarning):
    pass


class TorchStreamFallbackWarning(TorchStreamWarning):
    pass


FALLBACK_WARNING = (
    "The `{operation}` operation is not supported for `StreamTensors`{description}. "
    "Instead, the `{operation}` operation will return a regular `torch.Tensor` and subsequent "
    "operations will be performed without the use of the torchstream functionality. If this was "
    "expected, you can suppress this warning by calling `torchstream.suppress_warnings(True)` or by "
    "converting your input StreamTensor(s) to regular `torch.Tensors` manually before passing them to "
    "the `{operation}` operation. If this was unexpected it might have lead to an error being raised and "
    "might indicate an error in your code. If you got an error and think it's a bug in TorchStream, please "
    "open an issue on GitHub."
)

DESCRIPTION_METADATA_AMBIGUOUS = "since combining the StreamMetadata of the inputs is ambiguous"


def fallback_operation_warning(operation: str, description: str = ""):
    warnings.warn(
        FALLBACK_WARNING.format(operation=operation, description=" " + description if description else ""),
        TorchStreamFallbackWarning,
        stacklevel=2,
    )


# TODO (JDH): Add a test for this function
def suppress_warnings(suppress: bool = True):
    """Enable or disable warnings about fallback to regular torch operations."""
    warnings.simplefilter("ignore" if suppress else "default", TorchStreamWarning)
