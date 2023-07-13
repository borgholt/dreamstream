import functools
import itertools

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort


# https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557
# https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505
# https://pastebin.com/AkvAyJBw
# https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py


def as_metadata_tensor(
    x: torch.Tensor,
    metadata: dict,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> "MetadataTensor":
    x = torch.as_tensor(x, dtype=dtype, device=device)
    return MetadataTensor(x, metadata)


class MetadataTensor(torch.Tensor):
    tensor: torch.Tensor
    metadata: dict

    __slots__ = ["tensor", "metadata"]

    @staticmethod
    def __new__(cls, tensor, metadata: dict):
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

    def __init__(self, tensor, metadata: dict):
        super().__init__()
        self.metadata = metadata

    def numpy(self):
        return self.tensor.numpy()

    def __repr__(self):
        return self.tensor.__repr__() + f" ,\nmetadata={self.metadata}"

    @classmethod
    def to_tensor(cls, x: "MetadataTensor") -> torch.Tensor:
        return x.tensor if isinstance(x, cls) else x

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print(f"__torch_dispatch__({func=})")

        # Flatten arbitrary pytrees of tensors in args and kwargs
        args, args_tree_spec = torch.utils._pytree.tree_flatten(args)
        kwargs, kwargs_tree_spec = torch.utils._pytree.tree_flatten(kwargs)

        # Retrieve metadata from all MetadataTensors and check that they are the same
        metas = [x.metadata for x in itertools.chain(args, kwargs) if isinstance(x, cls)]
        if not all(m == metas[0] for m in metas):
            raise ValueError("All metadata must be the same")

        metadata = metas[0]

        # Convert all tensors to torch.Tensor and reconstruct pytrees
        args = torch.utils._pytree.tree_unflatten([cls.to_tensor(i) for i in args], args_tree_spec)
        kwargs = torch.utils._pytree.tree_unflatten([cls.to_tensor(i) for i in kwargs], kwargs_tree_spec)

        # Call the original function
        out = super(MetadataTensor, cls).__torch_dispatch__(func, types, args, kwargs)

        # Wrap the output in a MetadataTensor
        out = torch.utils._pytree.tree_map(functools.partial(as_metadata_tensor, metadata=metadata), out)
        return out

    # @classmethod
    # def __torch_function__(self, func, types, args=(), kwargs=None):
    #     print(f"__torch_function__({func=})")
    #     return super().__torch_function__(func, types, args, kwargs)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


if __name__ == "__main__":
    model = Model()
    scripted_model = torch.jit.script(model)

    t = torch.rand(2, 2)
    b = torch.tensor([1.0])
    m = as_metadata_tensor(
        t, metadata={"owner": "Ministry of Silly Walks", "bias": b}, dtype=torch.float32
    )

    o1 = m.clone()  # meta is not preserved for StreamTensor but it is here?
    o2 = m.transpose(0, 1)
    o3 = torch.rand_like(m)

    print("Printing!")
    print(m)

    print(o1)
    print(o2)
    print(o3)

    y = model(m)

    y_scripted = scripted_model(m)

    print("Regular output\n", y)
    print("Scripted output\n", y_scripted)

    # ONNX export
    torch.onnx.export(
        model=scripted_model,
        args=m,
        f="model.onnx",
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )

    # ONNX import
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    # ONNX inference
    ort_session = ort.InferenceSession("model.onnx")
    outputs = ort_session.run(output_names=["output"], input_feed={"input": m.numpy()}, run_options=None)
    print("ONNX output\n", outputs[0])

    import IPython
    IPython.embed(using=False)
    
    # TODO (JDH): Enable passing state to `model.forward` and to the ONNX runtime/graph
    # TODO (JDH): Test control flow
