import torch

class TensorSubClass(torch.Tensor):
    def __init__(self, x: torch.Tensor):
        super().__init__()
        self.x = x

    def myNumber(self):
        return 123456


def fn(x: TensorSubClass):
    return x.myNumber()

x = TensorSubClass(torch.rand((2, 3, 4)))
print("Eager: ", fn(x))

scripted_fn = torch.jit.script(fn)
print("Scripted: ", scripted_fn(x))
