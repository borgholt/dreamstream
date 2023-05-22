from random import randint
from uuid import uuid4

import torch
from torch import nn

from dreamstream.utils.flags import BATCH, LENGTH
from dreamstream.nn.utils import pad_full_sequence
from dreamstream.patches import patch_conv_1d
from dreamstream.data import OutputCollector

def random_chunks(full_length):
    chunks = []
    chunk_sum, remaining = 0, full_length
    while remaining > 0:
        chunks.append(min(randint(0, 200), remaining))
        chunk_sum = sum(chunks)
        remaining = full_length - chunk_sum
    return chunks

conv = nn.Conv1d(256, 128, 7, padding=3)
conv = patch_conv_1d(conv)

# TEST 1: Test with multiple streams of different lengths.
sequences = [torch.rand(256, randint(50, 2000)) for i in range(32)]
ids = [str(uuid4()) for i in range(32)]
targets = {_id: conv(s) for _id, s in zip(ids, sequences)}
batch = pad_full_sequence(sequences, names=("F", LENGTH), ids=ids).align_to("B", "F", "L")
chunks = random_chunks(batch.size("L"))

stream_output = OutputCollector()
conv.online()
for x in batch.split(chunks, dim=2):
    y = conv(x)
    stream_output.update(y)