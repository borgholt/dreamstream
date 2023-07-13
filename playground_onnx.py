from typing import Tuple, Union
import torch

import torch.nn as nn

from dreamstream.utils.timing import timeit


class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        return x + 2


@torch.jit.script
def state_handle_pre_hook(
    module, inputs: Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    if len(inputs) == 1:
        return inputs

    input, state = inputs

    return (input,)


def patch_module(module: nn.Module):
    module.register_forward_pre_hook(state_handle_pre_hook)


def run_timing():
    x = torch.randn(1, 1, 10)
    m = CustomModule()

    script_m = torch.jit.script(m)

    script_m(x)

    timeit("m(x)", print_results=True, globals=locals())
    timeit("script_m(x)", print_results=True, globals=locals())


def run_pre_hook_test():
    x = torch.randn(1, 1, 10)
    m = CustomModule()

    patch_module(m)

    m(x)

    # script_m = torch.jit.script(m)

    import IPython

    IPython.embed(using=False)


if __name__ == "__main__":
    # run_timing()

    run_pre_hook_test()
