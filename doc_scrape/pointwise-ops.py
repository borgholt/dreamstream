from glob import glob
import urllib.request
from datetime import datetime

import torch
from bs4 import BeautifulSoup
from tqdm import tqdm

#from dreamstream.utils.dummies import TestTensor
from dreamstream.utils.listloaders import get_tensor_attr
from dreamstream.tensor import recouple, inplace_recouple, TestTensor

#from ..tests import test_dict


tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]], dtype=torch.float32)
#tensor = torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]])
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

soup = BeautifulSoup(html_doc, 'html.parser')

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
    # if hasattr(torch, n + "_"):
    #     func_names.append(f"{n}_")
    # if hasattr(torch.Tensor, n + "_"):
    #     func_names.append(f"Tensor.{n}_")

no_valid_input_found = []
no_valid_output = []
valid_funcs = []
recouple_funcs = []
recouple_inplace_funcs = []

for func_name in tqdm(func_names):
    
    # these are handled manually (either decouple or customized)
    if "quantize" in func_name:
        continue
    if func_name.endswith("real") or func_name.endswith("imag"):
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
            if func_name.endswith("mvlgamma") or func_name == "Tensor.polygamma":
                inputs += [1]
            if func_name == "polygamma":
                inputs = [1] + inputs
            
            target_inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs] if is_inplace else inputs
            out = func(*target_inputs)
            input_valid = True
            break
        
        except Exception as e:
            if i == 3:
                no_valid_input_found.append(func_name)
                #import IPython; IPython.embed()
    
    if input_valid:
        
        test_inputs = [to_test(x) if isinstance(x, torch.Tensor) else x for x in inputs]
        
        test_out = recouple(func, *test_inputs, _tensor_type=TestTensor)
        compare_tensors(test_out, out)
        recouple_funcs.append(func_name)
        
        # try:
        #     test_out = valid(func, *test_inputs)
        #     compare_tensors(test_out, out)
        #     valid_funcs.append(func_name)
        #     print("\nSUCCESS\n")
        #     break
        # except:
        #     pass
        
        # try:
        #     if is_inplace:
        #         test_out = inplace_recouple(func, *test_inputs)
        #         compare_tensors(test_out, out)
        #         recouple_inplace_funcs.append(func_name)
        #     else:
        #         test_out = recouple(func, *test_inputs)
        #         compare_tensors(test_out, out)
        #         recouple_funcs.append(func_name)
        #     break
        # except:
        #     pass
        
        no_valid_output.append(func_name)

            
            
        
        
        
    #     try:
    #         stream_inputs = [to_test(x) for x in inputs]
    #         test_out = func(*stream_inputs)
    #     except Exception as e:
    #         test_dict[func_name] = str(e)
    #         continue

    #     if not isinstance(test_out, TestTensor):
    #         test_dict[func_name] = "NOT_TENSOR"
    #         continue
        
    #     if not hasattr(test_out, "meta"):
    #         test_dict[func_name] = "NO_META"
    #         continue
        
    #     if test_out.meta != "test":
    #         test_dict[func_name] = "NOT_VALID_META"
    #         continue
        
    #     nan_filter = torch.isnan(out)
    #     out = out[~nan_filter]
    #     test_out = torch.Tensor(test_out).rename(None)[~nan_filter]
        
    #     if torch.allclose(out, test_out):
    #         test_dict[func_name] = "VALID"
    #     else:
    #         test_dict[func_name] = "NOT_CLOSE"








# if len(glob("lists/pointwise-ops-*.txt")) > 0:
#     raise Exception("lists/pointwise-ops-*.txt already exists. Please delete this/these file(s) before running this script.")
# scrape_date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
# with open(f"lists/default-valid-pointwise-ops-{scrape_date}.txt", "w") as file_buffer:
#     file_buffer.write("\n".join(pwo_defualt_valid))
# with open(f"lists/valid-pointwise-ops-{scrape_date}.txt", "w") as file_buffer:
#     file_buffer.write("\n".join(pwo_valid))