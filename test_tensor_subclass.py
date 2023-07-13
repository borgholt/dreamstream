import torch

class TensorSubClass(torch.Tensor):
    def __init__(self, x: torch.Tensor):
        super().__init__()
        self.x = x

    def __len__(self):
        return 123456


def fn(x: TensorSubClass):
    return len(x)

x = TensorSubClass(torch.rand((2, 3, 4)))
length_from_eager = fn(x)

scripted_fn = torch.jit.script(fn)
length_from_jit = scripted_fn(x)

if length_from_eager != length_from_jit:
    print("Eager and jit behavior mismatching: %d vs %d" % (length_from_eager, length_from_jit))
else:
    print("Eager and jit behavior match, both are: %d" % length_from_jit)
