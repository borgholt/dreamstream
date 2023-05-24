import torch

from dreamstream.tensor import StreamTensor, StreamState
from dreamstream.utils.flags import LENGTH, BATCH

def rand_stream_tensor(*size, names, **kwargs):
    """
    Creates a StreamTensor with random values and metadata.
    """
    tensor = torch.rand(*size, **kwargs).rename(*names)
    ids = [str(i) for i in range(tensor.size(BATCH))]
    lengths = (torch.rand(tensor.size(BATCH)) * tensor.size(LENGTH)).to(torch.int64)
    is_first = torch.rand(tensor.size(BATCH)) < 0.5
    is_last = torch.rand(tensor.size(BATCH)) < 0.5
    state = StreamState(ids, is_first, is_last, lengths)
    return StreamTensor(tensor, state)