import torch
import torch.nn as nn
# import torchvision

import onnx
import onnxruntime as ort

from dreamstream.utils.timing import timeit


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model = nn.Transformer(nhead=8, num_encoder_layers=12, num_decoder_layers=6, dim_feedforward=2048, d_model=512, dropout=0.1)
model = nn.TransformerEncoder(nn.TransformerEncoderLayer(nhead=8, dim_feedforward=2048, d_model=512, dropout=0.1), num_layers=12)
model = model.to(device)
model.eval()


x = torch.randn(10, 32, 512)  # (seq, batch, dim)
tgt = torch.randn(20, 32, 512)

x = x.to(device)
tgt = tgt.to(device)

onnx_x = x.detach().cpu().numpy()
onnx_tgt = tgt.detach().cpu().numpy()


# model = torchvision.models.resnet18(pretrained=True)


# class SuperResolutionNet(nn.Module):
#     def __init__(self, upscale_factor, inplace=False):
#         super().__init__()

#         self.relu = nn.ReLU(inplace=inplace)
#         self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
#         self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
#         self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
#         self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.pixel_shuffle(self.conv4(x))
#         return x


# # Create the super-resolution model by using the above model definition.
# model = SuperResolutionNet(upscale_factor=3)
# model = model.to(device)

# x = torch.randn(1, 1, 224, 224, requires_grad=True)
# x = x.to(device)

# onnx_x = x.detach().cpu().numpy()

compiled_model = torch.compile(model)

scripted_model = torch.jit.script(model)

import IPython
IPython.embed(using=False)


# ONNX
torch.onnx.export(model, x, "transformer.onnx", verbose=True, opset_version=18, input_names=["input"], output_names=["output"])  # , operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
onnx_model = onnx.load("transformer.onnx")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

sess_options = ort.SessionOptions()
sess_options.enable_profiling = True
providers = ["CUDAExecutionProvider" if device.type == "cuda" else "CPUExecutionProvider"]
ort_session = ort.InferenceSession("transformer.onnx", providers=providers, sess_options=sess_options)

onnx_model = lambda x: ort_session.run(output_names=["output"], input_feed={"input": onnx_x}, run_options=None)

# onnx_model(onnx_x)
# onnx_model(onnx_x)
# onnx_model(onnx_x)
# onnx_model(onnx_x)
# onnx_model(onnx_x)
# onnx_model(onnx_x)


# .to("cpu").detach().numpy()
# .to("cpu").detach().numpy()
# .to("cpu").detach().numpy()
# .to("cpu").detach().numpy()
# .to("cpu").detach().numpy()
# .to("cpu").detach().numpy()

# x.to(device)
# x.to(device)
# x.to(device)
# x.to(device)
# x.to(device)
# x.to(device)
# x.to(device)


@torch.inference_mode(True)
def forward(model, x):
    return model(x)


timeit(lambda: forward(model, x), globals=locals(), print_results=True, print_suffix=f"model(x), device={device}")
timeit(lambda: forward(model, x), globals=locals(), print_results=True, print_suffix=f"model(x), device={device}")

timeit(lambda: forward(scripted_model, x), globals=locals(), print_results=True, print_suffix=f"scripted_model(x), device={device}")
timeit(lambda: forward(scripted_model, x), globals=locals(), print_results=True, print_suffix=f"scripted_model(x), device={device}")

timeit(lambda: forward(compiled_model, x), globals=locals(), print_results=True, print_suffix=f"compiled_model(x), device={device}")
timeit(lambda: forward(compiled_model, x), globals=locals(), print_results=True, print_suffix=f"compiled_model(x), device={device}")

# timeit(onnx_model(onnx_x), globals=locals(), print_results=True, print_suffix=f"onnx_model(x), device={device}")
# timeit(onnx_model(onnx_x), globals=locals(), print_results=True, print_suffix=f"onnx_model(x), device={device}")


"""
number=  500 | [   699µs,  703.1µs] |  700.7µs |  700.9µs +-  1.232µs | model(x), device=cuda
number=  500 | [ 702.9µs,  705.3µs] |  703.8µs |    704µs +-  796.5ns | model(x), device=cuda
number=  500 | [ 703.6µs,  706.3µs] |  705.2µs |  705.1µs +-  690.2ns | scripted_model(x), device=cuda
number=  500 | [ 705.5µs,  708.1µs] |  706.9µs |  706.8µs +-    929ns | scripted_model(x), device=cuda
number=  500 | [   580µs,  589.2µs] |  583.2µs |  583.6µs +-  2.959µs | compiled_model(x), device=cuda
number=  500 | [ 583.7µs,    586µs] |  584.8µs |  584.8µs +-  718.4ns | compiled_model(x), device=cuda
number=  200 | [ 1.165ms,  1.201ms] |   1.17ms |  1.175ms +-  11.76µs | onnx_model(x), device=cuda
number=  200 | [  1.17ms,  1.194ms] |  1.182ms |  1.182ms +-    7.4µs | onnx_model(x), device=cuda
"""
