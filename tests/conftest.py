import random
from typing import List

import pytest
import torch

from dreamstream.nn.utils.pad_sequence import pad_full_sequence, pad_stream_tensor
from dreamstream.tensor import StreamTensor
from dreamstream.utils.flags import BATCH, LENGTH


torch.manual_seed(42)
random.seed(42)


BATCH_SIZE = 32


def random_chunks(full_length, min_size: int = 1000, max_size: int = 8000):
    """Generate a number of random chunks of different lengths that sum up to `full_length`."""
    chunks = []
    chunk_sum, remaining = 0, full_length
    while remaining > 0:
        chunks.append(min(random.randint(min_size, max_size), remaining))
        chunk_sum = sum(chunks)
        remaining = full_length - chunk_sum
    return chunks


@pytest.fixture
def sequences():
    """A number of sequences of different lengths of size `BATCH_SIZE`. Size (16, L) of `torch.rand` values."""
    return [torch.rand(16, random.randint(200, 1000)) for i in range(BATCH_SIZE)]


@pytest.fixture
def ids():
    """A number of unique ids of size `BATCH_SIZE`."""
    return [str(hash(i)) for i in range(BATCH_SIZE)]


@pytest.fixture
def batches_of_chunks(sequences, ids) -> List[StreamTensor]:
    """Batches of chunks of different lengths from the `sequences` data."""
    data = pad_full_sequence(sequences, names=("F", LENGTH), ids=ids).align_to(BATCH, "F", LENGTH)
    data = data.unpad_sequence()
    data = {
        _id: s.split(random_chunks(s.size(LENGTH), min_size=100, max_size=200), dim=1) for _id, s in zip(ids, data)
    }

    def num_remaining_chunks(data):
        return sum([len(x) for x in data.values()])

    batches = []
    while num_remaining_chunks(data) > 0:
        batch = [s.pop(0) for _id, s in data.items() if len(s) > 0 and random.random() < 0.75]
        if len(batch) > 0:
            batches.append(pad_stream_tensor(batch).align_to(BATCH, "F", LENGTH))

    return batches
