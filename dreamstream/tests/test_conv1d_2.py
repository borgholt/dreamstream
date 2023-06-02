from random import randint
from uuid import uuid4
import random

import torch
from torch import nn

from dreamstream.utils.flags import LENGTH
from dreamstream.nn.utils import pad_full_sequence, pad_stream_tensor
from dreamstream.patches import patch_conv_1d
from dreamstream.data import OutputCollector


def random_chunks(full_length):
    chunks = []
    chunk_sum, remaining = 0, full_length
    while remaining > 0:
        chunks.append(min(randint(7-3, 100), remaining))
        chunk_sum = sum(chunks)
        remaining = full_length - chunk_sum
    return chunks


conv = nn.Conv1d(256, 128, 7, stride=6, padding=3)
conv = patch_conv_1d(conv)

# TEST 1: Test with multiple streams of different lengths.
sequences = [torch.rand(256, randint(50, 2000)) for i in range(32)]

ids = [str(uuid4()) for i in range(32)]
targets = {_id: conv(s) for _id, s in zip(ids, sequences)}
original_sequences = {_id: s for _id, s in zip(ids, sequences)}

data = pad_full_sequence(sequences, names=("F", LENGTH), ids=ids).align_to("B", "F", "L")
data = data.unpad_sequence()
data = {_id: s.split(random_chunks(s.size("L")), dim=1) for _id, s in zip(ids, data)}

def remaining_chunks(data):
    return sum([len(x) for x in data.values()])

batches = []
while remaining_chunks(data) > 0:
    batch = [s.pop(0) for _id, s in data.items() if len(s) > 0 and random.random() < 0.75]
    if len(batch) > 0:
        batches.append(pad_stream_tensor(batch).align_to("B", "F", "L"))

stream_output = OutputCollector()
conv.online()
for x in batches:
    y = conv(x)
    stream_output.update(y)
    
for _id, _y in targets.items():
    y = stream_output[_id].tensor()
    abs_diff = (_y - y).abs()
    print(y.size(-1), torch.allclose(_y, y), abs_diff.max().item(), abs_diff.max(0).values[:10].max().item())
