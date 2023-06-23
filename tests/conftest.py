import random

import pytest
import torch

from dreamstream.nn.utils.pad_sequence import pad_full_sequence, pad_stream_tensor
from dreamstream.utils.flags import BATCH, LENGTH


torch.manual_seed(42)
random.seed(42)


BATCH_SIZE = 8
SAMPLE_RATE = 1000
WAVEFORM_MIN_SECONDS = 3
WAVEFORM_MAX_SECONDS = 7
WAVEFORM_CHUNK_MIN_SECONDS = 1
WAVEFORM_CHUNK_MAX_SECONDS = 2


def random_chunks(full_length, min_size: int = 1000, max_size: int = 8000):
    chunks = []
    chunk_sum, remaining = 0, full_length
    while remaining > 0:
        chunks.append(min(random.randint(min_size, max_size), remaining))
        chunk_sum = sum(chunks)
        remaining = full_length - chunk_sum
    return chunks


@pytest.fixture
def waveforms():
    """A number of waveform sequences of different lengths of size `BATCH_SIZE`. Size (16, L) of `torch.rand` values."""
    return [
        torch.rand(1, random.randint(SAMPLE_RATE * WAVEFORM_MIN_SECONDS, SAMPLE_RATE * WAVEFORM_MAX_SECONDS))
        for i in range(BATCH_SIZE)
    ]


@pytest.fixture
def ids():
    """A number of unique ids of size `BATCH_SIZE`."""
    return [str(hash(i)) for i in range(BATCH_SIZE)]


@pytest.fixture
def batches_of_waveform_chunks(waveforms, ids):
    """Batches of chunks of waveforms of varying lengths from the `waveforms` data."""
    data = pad_full_sequence(waveforms, names=("F", LENGTH), ids=ids).align_to("B", "F", "L")
    data = data.unpad_sequence()
    data = {
        _id: s.split(
            random_chunks(
                s.size("L"),
                min_size=SAMPLE_RATE * WAVEFORM_CHUNK_MIN_SECONDS,
                max_size=SAMPLE_RATE * WAVEFORM_CHUNK_MAX_SECONDS,
            ),
            dim=1,
        )
        for _id, s in zip(ids, data)
    }

    def num_remaining_chunks(data):
        return sum([len(x) for x in data.values()])

    batches = []
    while num_remaining_chunks(data) > 0:
        batch = [s.pop(0) for _id, s in data.items() if len(s) > 0 and random.random() < 0.75]
        if len(batch) > 0:
            batches.append(pad_stream_tensor(batch).align_to(BATCH, "F", LENGTH))

    return batches
