from glob import glob
import urllib.request
from datetime import datetime
from copy import deepcopy

import torch
from bs4 import BeautifulSoup
from tqdm import tqdm

# from dreamstream.utils.dummies import TestTensor
from dreamstream.utils.listloaders import get_tensor_attr
from dreamstream.tensor import recouple, inplace_recouple, TestTensor

# from ..tests import test_dict


tensor = torch.tensor(
    [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]], dtype=torch.float32
)
# tensor = torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]])
tensors = [tensor.clone() for i in range(3)]


def to_test(tensor):
    return TestTensor(tensor.rename("A", "B", "C"), meta="test")


class Inputs:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter((self.args, self.kwargs))


def valid(func, *args, **kwargs):
    return func(*args, **kwargs)


def default_valid(func, *args, **kwargs):
    out = func(*args, **kwargs)

    metas = [x.meta for x in [*args, *kwargs.values()] if isinstance(x, TestTensor)]
    if not all(s == metas[0] for s in metas[1:]):
        msg = (
            f"Called a torch function ({func.__name__}) which was not handled by "
            f"StreamTensor.__torch_function__ with {len(metas)} StreamTensors in the input."
            f"In this case the function can only be handled if the StreamTensors have equal metadata,"
            f"but they were not equal."
        )
        raise RuntimeError(msg)

    if isinstance(out, TestTensor):
        out.meta = metas[0]
        return out
    elif isinstance(out, torch.Tensor):
        return TestTensor(out, meta=metas[0])

    return out


def compare_tensors(test_out, out):
    nan_filter = torch.isnan(out)
    out = out[~nan_filter]

    assert out.numel() > 0
    assert isinstance(test_out, TestTensor)
    assert hasattr(test_out, "meta")
    assert test_out.meta == "test"
    assert torch.allclose(torch.Tensor(test_out).rename(None)[~nan_filter], out)


inplace_func_hierarchy = [valid, inplace_recouple]
outofplace_func_hierarchy = [valid, recouple]

fp = urllib.request.urlopen("https://pytorch.org/docs/stable/torch.html")
html_doc = fp.read().decode("utf8")
fp.close()

soup = BeautifulSoup(html_doc, "html.parser")

# scrape and validate pointwise ops
pwo_section = soup.find("section", {"id": "pointwise-ops"})
pwo_rows = pwo_section.find_all("tr")
pwo_names = [r.find("td").text for r in pwo_rows]

func_names = []
for n in pwo_names:
    if hasattr(torch, n):
        func_names.append(f"{n}")
    if hasattr(torch.Tensor, n):
        func_names.append(f"Tensor.{n}")
    if hasattr(torch, n + "_"):
        func_names.append(f"{n}_")
    if hasattr(torch.Tensor, n + "_"):
        func_names.append(f"Tensor.{n}_")

no_valid_input_found = []
no_valid_output = []
valid_funcs = []
default_valid_funcs = []
recouple_funcs = []
recouple_inplace_funcs = []

for func_name in tqdm(func_names):
    # these are handled manually (either decouple or customized)
    if "quantize" in func_name:
        continue
    if func_name.endswith("real") or func_name.endswith("imag"):
        continue
    if func_name.endswith("frexp"):  # or func_name.endswith("ldexp"):
        continue
    if func_name.endswith("gradient"):
        continue

    is_inplace = func_name.endswith("_")
    func = get_tensor_attr(func_name)
    input_valid = False

    for i in range(1, 4):
        try:
            inputs = tensors[:i]

            if "bitwise" in func_name:
                inputs = [x.to(torch.int64) for x in inputs]
            if func_name.endswith("softmax"):
                inputs += [-1]
            if ("mvlgamma" in func_name) or ("Tensor.polygamma" in func_name):
                inputs += [1]
            if func_name == "polygamma":
                inputs = [1] + inputs
            if func_name.endswith("float_power_"):
                inputs = [x.to(torch.float64) for x in inputs]

            target_inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs] if is_inplace else inputs
            out = func(*target_inputs)
            input_valid = True
            break

        except Exception as e:
            if i == 3:
                no_valid_input_found.append(func_name)

    if input_valid:
        try:
            if is_inplace:
                test_inputs = [to_test(x.clone()) if isinstance(x, torch.Tensor) else deepcopy(x) for x in inputs]
                test_out = valid(func, *test_inputs)
                compare_tensors(test_out, out)
                valid_funcs.append(func_name)
            else:
                test_inputs = [to_test(x) if isinstance(x, torch.Tensor) else x for x in inputs]
                test_out = default_valid(func, *test_inputs)
                compare_tensors(test_out, out)
                default_valid_funcs.append(func_name)
            continue
        except Exception as e:
            pass

        try:
            if is_inplace:
                test_inputs = [to_test(x.clone()) if isinstance(x, torch.Tensor) else deepcopy(x) for x in inputs]
                test_out = inplace_recouple(func, *test_inputs, _tensor_type=TestTensor)
                compare_tensors(test_out, out)
                recouple_inplace_funcs.append(func_name)
            else:
                test_inputs = [to_test(x) if isinstance(x, torch.Tensor) else x for x in inputs]
                test_out = recouple(func, *test_inputs, _tensor_type=TestTensor)
                compare_tensors(test_out, out)
                recouple_funcs.append(func_name)
            continue
        except Exception as e:
            pass

        no_valid_output.append(func_name)

print(f"\n\nNumber of valid funcs: {len(valid_funcs)}")
print(f"Number of default valid funcs: {len(default_valid_funcs)}")
print(f"Number of recouple funcs: {len(recouple_funcs)}")
print(f"Number of inplace recouple funcs: {len(recouple_inplace_funcs)}")
print(f"Number of funcs wo/ valid input: {len(no_valid_input_found)}")
print(f"Number of funcs wo/ valid output: {len(no_valid_output)}")


if len(glob("lists/*pointwise-ops*.txt")) > 0:
    raise Exception(
        "lists/*pointwise-ops*.txt already exists. Please delete this/these file(s) before running this script."
    )
scrape_date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
with open(f"lists/default-valid-pointwise-ops-{scrape_date}.txt", "w") as file_buffer:
    file_buffer.write("\n".join(default_valid_funcs))
with open(f"lists/valid-pointwise-ops-{scrape_date}.txt", "w") as file_buffer:
    file_buffer.write("\n".join(valid_funcs))
with open(f"lists/recouple-pointwise-ops-{scrape_date}.txt", "w") as file_buffer:
    file_buffer.write("\n".join(recouple_funcs))
with open(f"lists/inplace-recouple-pointwise-ops-{scrape_date}.txt", "w") as file_buffer:
    file_buffer.write("\n".join(recouple_inplace_funcs))
