from typing import Any, Union

import torch


def linear_override(input, weight, bias=None):
    print("Calling linear override")
    x = torch.nn.functional.linear(input, weight, bias)
    return x + 10.0


OVERRIDES = {
    torch.nn.functional.linear: linear_override,
}


class TensorSubClass(torch.Tensor):
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.tensor = tensor

    @classmethod
    def to_tensor(cls, input: Any) -> Union[torch.Tensor, Any]:
        return input.tensor if isinstance(input, TensorSubClass) else input

    def tolist(self) -> list:
        return self.tensor.tolist()

    def numpy(self) -> Any:
        return self.tensor.numpy()

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print(f"Calling __torch_dispatch__ for {func} with name '{func.__name__}'")
        if kwargs is None:
            kwargs = {}

        if func not in OVERRIDES:
            return super().__torch_dispatch__(func, types, args, kwargs)

        print(f"Calling function override for {func}")

        # To avoid infinite recursion we first convert all tensors to torch.Tensor. 
        args = torch.utils._pytree.tree_map(cls.to_tensor, args)
        kwargs = torch.utils._pytree.tree_map(cls.to_tensor, kwargs)

        # Call override, convert tensors to TensorSubClass and reconstruct pytrees.
        out = OVERRIDES[func](*args, **kwargs)
        return torch.utils._pytree.tree_map(cls, out)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: TensorSubClass):
        return self.sigmoid(self.linear(x).sum(-1))


torch.manual_seed(0)

model = Model()
scripted_model = torch.jit.script(model)

x = TensorSubClass(torch.rand((2, 4)))

print("\n=== Executing eager model ===")
out_from_eager = model(x)
# Prints `TensorSubClass([1., 1.], grad_fn=<AliasBackward0>)`
print("Eager: ", out_from_eager)

print("\n=== Executing scripted model ===")
out_from_scripted = scripted_model(x)
# Prints `TensorSubClass([0.5440, 0.4919], grad_fn=<AliasBackward0>)`
print("Scripted: ", out_from_scripted, "\n")
