import types

import torch.nn as nn


def online(self: nn.Module, mode: bool = True) -> nn.Module:
    """Set the module to either online (default) or offline mode."""
    if not isinstance(mode, bool):
        raise ValueError(f"Streaming `mode` was expected to be boolean but got {mode=}.")

    self.streaming = mode
    for module in self.children():
        module.online(mode)
    return self


def offline(self: nn.Module) -> nn.Module:
    """Set the module to offline mode."""
    return self.online(mode=False)


def add_streaming_modes(module: nn.Module):
    """Equip a module with the streaming mode but no additional functionality."""
    module.online = types.MethodType(online, module)
    module.offline = types.MethodType(offline, module)
    module.streaming = False
    return module
