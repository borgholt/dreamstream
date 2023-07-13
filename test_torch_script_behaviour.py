import torch


class TensorSubClass(torch.Tensor):
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.tensor = tensor

    def __len__(self) -> int:
        """Modify the behaviour of len() to return a constant value."""
        return 123456

    def new_method(self) -> int:
        """Add a new method to the subclass that does not exist on `torch.Tensor`."""
        return 123456


def call_length(tensor: TensorSubClass) -> int:
    return len(tensor)


def call_new_method(tensor: TensorSubClass) -> int:
    return tensor.new_method()


torch.manual_seed(0)

call_length_scripted = torch.jit.script(call_length)

tensor = TensorSubClass(torch.rand((2, 3, 4)))

length_from_eager = call_length(tensor)
print("Eager: ", length_from_eager)  # Prints 123456
length_from_jit = call_length_scripted(tensor)
print("Scripted: ", length_from_jit)  # Prints 2

call_new_method_scripted = torch.jit.script(call_new_method)  # Throws RuntimeError


# new_method_from_eager = call_new_method(x)
# print("Eager: ", new_method_from_eager)
# try:
#     new_method_from_scripted = call_new_method_scripted(x)
# except Exception as e:
#     print("Scripted: ", e)
