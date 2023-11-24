import functools
import itertools
from typing import Dict, Optional

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import rich


# https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557
# https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505
# https://pastebin.com/AkvAyJBw
# https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py


def as_metadata_tensor(
    x: torch.Tensor,
    metadata: dict,
    state: Optional[Dict[str, torch.Tensor]] = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> "MetadataTensor":
    x = torch.as_tensor(x, dtype=dtype, device=device)
    return MetadataTensor(x, metadata, state)


def linear_override(input, weight, bias=None, state: Dict[str, torch.Tensor] = None):
    print(input)
    output = torch.nn.functional.linear(input.tensor, weight, bias)
    output = output + input.state["linear_1_extra_bias"]
    return as_metadata_tensor(output, input.metadata, input.state)


OVERRIDES = {
    torch.nn.functional.linear: linear_override,
}


# ATEN_OVERRIDES = {
# }


class MetadataTensor(torch.Tensor):
    tensor: torch.Tensor
    metadata: dict

    __slots__ = ["tensor", "metadata"]

    @staticmethod
    def __new__(cls, tensor, metadata: dict, state: Optional[Dict[str, torch.Tensor]] = None):
        # The wrapping tensor (MetadataTensor) shouldn't hold any
        # memory for the class in question, but it should still
        # advertise the same device as before.
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            # TODO: clone storage aliasing
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
        )
        # We reference the underlying TorchTensor via an attribute on the MetadataTensor to allow unwrapping it
        # in __torch_dispatch__ without recursion.
        r.tensor = tensor
        return r

    def __init__(self, tensor, metadata: dict, state: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.metadata = metadata
        self.state = state

    def set_state(self, state: dict):
        self.state = state

    def get_state(self):
        return self.state

    def numpy(self):
        return self.tensor.numpy()

    def __repr__(self):
        return self.tensor.__repr__() + f" ,\nmetadata={self.metadata},\nstate={self.state}"

    @classmethod
    def to_tensor(cls, x: "MetadataTensor") -> torch.Tensor:
        return x.tensor if isinstance(x, cls) else x

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        rich.print(f"__torch_dispatch__({func=})")
        # import IPython
        # IPython.embed(using=False)
        
        # Flatten arbitrary pytrees of tensors in args and kwargs.
        args, args_tree_spec = torch.utils._pytree.tree_flatten(args)
        kwargs, kwargs_tree_spec = torch.utils._pytree.tree_flatten(kwargs)

        # Retrieve all metadatas and states.
        metas, states = list(zip(*[(x.metadata, x.state) for x in itertools.chain(args, kwargs) if isinstance(x, cls)]))

        # Convert all tensors to torch.Tensor and reconstruct pytrees.
        args = torch.utils._pytree.tree_unflatten([cls.to_tensor(i) for i in args], args_tree_spec)
        kwargs = torch.utils._pytree.tree_unflatten([cls.to_tensor(i) for i in kwargs], kwargs_tree_spec)

        # Call the original function.
        out = super(MetadataTensor, cls).__torch_dispatch__(func, types, args, kwargs)

        # Wrap the output in a MetadataTensor.
        out = torch.utils._pytree.tree_map(functools.partial(as_metadata_tensor, metadata=metas[0], state=states[0]), out)
        return out

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        rich.print(f"__torch_function__({func=}, {func.__name__})")
        if kwargs is None:
            kwargs = {}

        if func in OVERRIDES:
            print(f"Calling function override for {func}")
            return OVERRIDES[func](*args, **kwargs)

        return super().__torch_function__(func, types, args, kwargs)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        # self.conv = nn.Conv1d(2, 2, 3)
        self.sigmoid = nn.Sigmoid()
        # self.lstm = nn.LSTM(2, 2, batch_first=True)
        # self.transformer_encoder_layer = nn.TransformerEncoderLayer(2, 2, batch_first=True)

    def user_defined_forward(self, x):
        x = self.linear(x)  # (B, T, C)
        # x = x.permute(0, 2, 1)  # (B, C, T)
        # x = self.conv(x)
        # x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.sigmoid(x)
        # x, c = self.lstm(x)  # (B, T, C)
        # x = self.transformer_encoder_layer(x)  # (B, T, C)
        return x

    def forward(self, x: MetadataTensor, _state: Optional[Dict[str, torch.Tensor]] = None):
        if _state is not None:  # TODO (JDH): This should go in an outer level wrapper somehow.
            x.set_state(_state)

        y = self.user_defined_forward(x)

        # TODO (JDH): This should go in an outer level wrapper somehow.
        _state = y.get_state()
        print(_state)

        return y, _state


if __name__ == "__main__":
    torch.manual_seed(0)
    
    model = Model()
    scripted_model = torch.jit.script(model)

    t = torch.rand(2, 7, 2)  # (B, T, C)
    b = torch.tensor([10.0])
    m = as_metadata_tensor(
        t, metadata={"owner": "Ministry of Silly Walks"}, state={"linear_1_extra_bias": b}, dtype=torch.float32
    )

    # o1 = m.clone()  # meta is not preserved for StreamTensor but it is here?
    # o2 = m.transpose(0, 1)
    # o3 = torch.rand_like(m)

    # print("Printing!")
    # print(m)

    # print(o1)
    # print(o2)
    # print(o3)

    print("Running model")
    y = model(m)

    print("Running scripted model")
    y_scripted = scripted_model(m)

    print("Regular output\n", y)
    print("Scripted output\n", y_scripted)

    import IPython
    IPython.embed(using=False)
    
    exit(0)

    # ONNX export
    torch.onnx.export(
        model=scripted_model,
        args=(m, m.state),
        f="model.onnx",
        export_params=True,
        opset_version=18,
        input_names=["input", "state"],
        output_names=["output", "state"],
    )

    # ONNX import
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    # ONNX inference
    ort_session = ort.InferenceSession("model.onnx")
    outputs = ort_session.run(output_names=["output"], input_feed={"input": m.numpy()}, run_options=None)
    print("ONNX output\n", outputs[0])

    # TODO (JDH): Enable passing state to `model.forward` and to the ONNX runtime/graph
    # TODO (JDH): Test control flow