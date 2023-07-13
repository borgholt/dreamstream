import functools
import itertools
import collections

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort


# https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557
# https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505
# https://pastebin.com/AkvAyJBw
# https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py


Metadata = collections.namedtuple("Metadata", ["bias"])

State = Tuple[torch.Tensor, torch.Tensor]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)
        self.sigmoid = nn.Sigmoid()
        
        self.state_bias = torch.tensor([2.0])
        self.state_buffer  = torch.empty((2, 3))

    # def forward(
    #     self, input: torch.Tensor, metadata: Dict[str, torch.Tensor], state: Dict[str, torch.Tensor]
    # ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # x = self.linear(input)
    # x = self.sigmoid(x + metadata["bias"])
    # x = x + state["bias"]
    # state["bias"] = state["bias"] + torch.tensor([1.0])
    # return x, state

    def get_state(self) -> Optional[State]:
        if len(self.state_buffer) > 0:
            return (self.state_buffer, self.state_bias)
        return None

    def set_state(self, state: State):
        self.state_buffer, self.state_index, self.stride_index = state

    def forward(
        self,
        input: torch.Tensor,
        # metadata: torch.Tensor,
    ) -> torch.Tensor:
        x = self.linear(input) + self.state_bias
        x = self.sigmoid(x)
        self.state_buffer = torch.cat([self.state_buffer + x], dim=0)
        self.state_bias = self.state_bias + torch.tensor([1.0])
        return x


if __name__ == "__main__":
    model = Model()
    scripted_model = torch.jit.script(model)

    t = torch.rand(2, 2)
    b1 = torch.tensor([3.0])
    b2 = torch.tensor([3.0])

    # metadata = Metadata(b1)
    # metadata = {"bias": b1}
    # state = {"bias": b2}
    metadata = b1
    state = b2

    print(model.state_bias)
    y = model(t) #, metadata, state)
    print(model.state_bias)

    print(scripted_model.state_bias)
    y_scripted = scripted_model(t) #, metadata, state)
    print(scripted_model.state_bias)

    print("Regular output\n", y)
    print("Scripted output\n", y_scripted)

    # ONNinput export
    torch.onnx.export(
        model=scripted_model,
        args=(t,),
        # args=(t, metadata, {}),
        f="model.onnx",
        # export_params=True,
        # opset_version=11,
        # input_names=["input", "metadata", "state"],
        input_names=["input"],
        output_names=["output"],
    )

    # ONNX import
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    # ONNX inference
    t_onnx = t.numpy()
    # metadata_onnx = {"bias": b1.numpy()}
    # state_onnx = {"bias": b2.numpy()}
    metadata_onnx = b1.numpy()
    state_onnx = b2.numpy()
    input_feed = {"input": t_onnx}#, "state": state_onnx}

    ort_session = ort.InferenceSession("model.onnx")
    outputs = ort_session.run(output_names=["output"], input_feed=input_feed, run_options=None)
    print("ONNX output\n", outputs[0])
    outputs = ort_session.run(output_names=["output"], input_feed=input_feed, run_options=None)
    print("ONNX output\n", outputs[0])

    import IPython
    IPython.embed(using=False)

    # TODO (JDH): Enable passing state to `model.forward` and to the ONNX runtime/graph
    # TODO (JDH): Test control flow
