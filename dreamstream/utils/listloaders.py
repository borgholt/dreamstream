import os
from glob import glob

import torch


lists_path = __file__.replace("/dreamstream/utils/listloaders.py", "/doc_scrape/lists")

def get_tensor_attr(x):
    return getattr(torch.Tensor, x.replace("Tensor.", "")) if x.startswith("Tensor.") else getattr(torch, x)

def load_default_valid_pointwise_ops():
    with open(glob(os.path.join(lists_path, "default-valid-pointwise-ops-*.txt"))[0], "r") as file_buffer:
        ops_list = {get_tensor_attr(f) for f in file_buffer.read().split("\n")}
    return ops_list

def load_valid_pointwise_ops():
    with open(glob(os.path.join(lists_path, "valid-pointwise-ops-*.txt"))[0], "r") as file_buffer:
        ops_list = {get_tensor_attr(f) for f in file_buffer.read().split("\n")}
    return ops_list

def load_recouple_pointwise_ops():
    with open(glob(os.path.join(lists_path, "recouple-pointwise-ops-*.txt"))[0], "r") as file_buffer:
        ops_list = {get_tensor_attr(f) for f in file_buffer.read().split("\n")}
    return ops_list

def load_inplace_recouple_pointwise_ops():
    with open(glob(os.path.join(lists_path, "inplace-recouple-pointwise-ops-*.txt"))[0], "r") as file_buffer:
        ops_list = {get_tensor_attr(f) for f in file_buffer.read().split("\n")}
    return ops_list

