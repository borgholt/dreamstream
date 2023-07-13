import functools
import inspect
import types

from typing import Any, Callable, Tuple

import rich
import torch
import torch.nn as nn


DEFAULT_GLOBAL_STATE = dict()

def get_global_state():
    return DEFAULT_GLOBAL_STATE


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


def dummy_pre_forward(module, inputs: Tuple[Any], state: dict):
    rich.print(f"pre_forward: {module=}, {inputs=}, {state=}")


def dummy_post_forward(module, inputs: Tuple[Any], outputs: Tuple[Any], state: dict):
    rich.print(f"post_forward: {module=}, {inputs=}, {outputs=}, {state=}")
    

def _patch_module(module: nn.Module, pre_forward: Callable[[Tuple[Any], dict], Tuple[Any]], post_forward: Callable[[Tuple[Any], dict], Tuple[Any]]):
    original_forward = module.forward
    module.original_forward = original_forward

    def wrapped_forward(self, *inputs, state: dict, **kwargs):
        """Augmented self forward method with state, pre_forward and post_forward arguments. 
        
        First calls the `pre_forward` function with the arguments and the state, then calls the original forward method, 
        and finally calls the `post_forward` function with the arguments and the state. The state is passed as a dictionary
        to allow for arbitrary state to be passed between the functions.
        """
        import IPython
        IPython.embed(using=False)
        if self.streaming:
            inputs, state = pre_forward(self, inputs, state)
            outputs = self.original_forward(self, *inputs, state, **kwargs)
            outputs, state = post_forward(self, inputs, outputs, state)
            return outputs, state

        return self.original_forward(self, *inputs, **kwargs)

    # functools.wraps(original_forward)(wrapped_forward)
    # functools.update_wrapper(wrapped_forward, original_forward)
    module.forward = wrapped_forward
    return module


MODULE_PATCHERS = {nn.Linear: functools.partial(_patch_module, pre_forward=dummy_pre_forward, post_forward=dummy_post_forward)}


def patch_module(module) -> None:
    """Patch a given module to support streaming mode."""
    patch_method = MODULE_PATCHERS.get(type(module), None)
    add_streaming_modes(module)
    if patch_method is not None:
        patch_method(module)


def patch(module: nn.Module):
    """Recursively patch a module and its children to support streaming mode."""
    module.apply(patch_module)


if __name__ == "__main__":
    m = nn.Linear(10, 10)
    patch(m)
    
    x = torch.randn(10)
    state = get_global_state()
    o = m(x, state=state, pre_forward=dummy_pre_forward, post_forward=dummy_post_forward)
