import torch.nn as nn

from dreamstream.patches.conv import patch_conv_1d
from dreamstream.patches.rnn import patch_rnn
from dreamstream.patches.modes import add_streaming_modes


MODULE_PATCHERS = {
    nn.Conv1d: patch_conv_1d,
    nn.RNN: patch_rnn,
    nn.LSTM: patch_rnn,
    nn.GRU: patch_rnn,
}


def patch(module: nn.Module) -> nn.Module:
    """Recursively apply `patch_module` to all modules in `module`."""
    module.apply(patch_module)
    return module


def patch_module(module) -> None:
    """Patch a given module to support streaming mode."""
    patch_method = MODULE_PATCHERS.get(type(module), None)
    if patch_method is None:
        add_streaming_modes(module)
    else:
        patch_method(module)

    # Error checking for modules that do NOT have the correct behaviour yet.
    if isinstance(
        module,
        (
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            # nn.LSTM,
            # nn.GRU,
            # nn.RNN,
            nn.MultiheadAttention,
        ),
    ):
        raise NotImplementedError(f"Module {type(module)} is not supported yet.")

    return module
