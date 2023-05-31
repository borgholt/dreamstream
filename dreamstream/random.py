import torch

from dreamstream.tensor import StreamTensor, StreamMetadata
from dreamstream.utils.flags import LENGTH, BATCH

def rand_stream_tensor(*size, names, **kwargs):
    """
    Creates a StreamTensor with random values and metadata.
    """
    tensor = torch.rand(*size, **kwargs).rename(*names)
    ids = [str(i) for i in range(tensor.size(BATCH))]
    lengths = (torch.rand(tensor.size(BATCH)) * tensor.size(LENGTH)).to(torch.int64)
    sos = torch.rand(tensor.size(BATCH)) < 0.5
    eos = torch.rand(tensor.size(BATCH)) < 0.5
    meta = StreamMetadata(ids, sos, eos, lengths)
    return StreamTensor(tensor, meta)