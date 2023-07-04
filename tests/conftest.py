import random
from typing import List

import pytest
import torch

from dreamstream.nn.utils.pad_sequence import pad_full_sequence, pad_stream_tensor
from dreamstream.utils.flags import BATCH, LENGTH


SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


BATCH_SIZE = 8

# Test parameters for waveforms
SAMPLE_RATE = 1000
WAVEFORM_DIM = 1  # Number of waveform channels
WAVEFORM_MIN_SECONDS = 3
WAVEFORM_MAX_SECONDS = 7
WAVEFORM_CHUNK_MIN_SECONDS = 1
WAVEFORM_CHUNK_MAX_SECONDS = 2
WAVEFORM_CHUNK_SECONDS = 1.5
WAVEFORM_MIN_SIZE = round(WAVEFORM_MIN_SECONDS * SAMPLE_RATE)
WAVEFORM_MAX_SIZE = round(WAVEFORM_MAX_SECONDS * SAMPLE_RATE)
WAVEFORM_CHUNK_MIN_SIZE = round(WAVEFORM_CHUNK_MIN_SECONDS * SAMPLE_RATE)
WAVEFORM_CHUNK_MAX_SIZE = round(WAVEFORM_CHUNK_MAX_SECONDS * SAMPLE_RATE)
WAVEFORM_CHUNK_SIZE = round(WAVEFORM_CHUNK_SECONDS * SAMPLE_RATE)

# Test parameters for sequences of "tokens" (shorter than waveforms)
# TOKEN_DIM = 4
# TOKEN_MIN_SIZE = 50
# TOKEN_MAX_SIZE = 500
# TOKEN_CHUNK_MIN_SIZE = 10
# TOKEN_CHUNK_MAX_SIZE = 30
# TOKEN_CHUNK_SIZE = 20
TOKEN_DIM = 4
TOKEN_MIN_SIZE = 50
TOKEN_MAX_SIZE = 500
TOKEN_CHUNK_MIN_SIZE = 10
TOKEN_CHUNK_MAX_SIZE = 30
TOKEN_CHUNK_SIZE = 20


def random_chunks(full_length, min_size: int = 1000, max_size: int = 8000) -> List[int]:
    """Return a list of chunk sizes to use for `torch.split`."""
    chunks = []
    chunk_sum, remaining = 0, full_length
    while remaining > 0:
        chunks.append(min(random.randint(min_size, max_size), remaining))
        chunk_sum = sum(chunks)
        remaining = full_length - chunk_sum
    return chunks


@pytest.fixture
def waveforms():
    """A `BATCH_SIZE` list of length-varying waveform sequences of size (WAVEFORM_DIM, L) of `torch.rand` values."""
    return [torch.rand(WAVEFORM_DIM, random.randint(WAVEFORM_MIN_SIZE, WAVEFORM_MAX_SIZE)) for _ in range(BATCH_SIZE)]


@pytest.fixture
def tokens():
    """A `BATCH_SIZE` list of length-varying token sequences of size (TOKEN_DIM, L) of `torch.rand` values."""
    return [torch.rand(TOKEN_DIM, random.randint(TOKEN_MIN_SIZE, TOKEN_MAX_SIZE)) for _ in range(BATCH_SIZE)]


@pytest.fixture
def ids():
    """A number of unique ids of size `BATCH_SIZE`."""
    return [str(hash(i)) for i in range(BATCH_SIZE)]


def create_random_batches(data, ids, min_size: int, max_size: int):
    """Create a number of batches of shape (BATCH, F, RANDOM_CHUNK_SIZE) from variable length sequences (F, LENGTH)."""
    data = pad_full_sequence(data, names=("F", LENGTH), ids=ids).align_to("B", "F", "L")
    data = data.unpad_sequence()
    data = {
        _id: s.split(random_chunks(s.size("L"), min_size=min_size, max_size=max_size), dim=1)
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


def create_structured_batches(data, ids, chunk_size: int):
    """Create a number of batches of shape (BATCH, F, CHUNK_SIZE) from variable length sequences (F, LENGTH)."""
    data = pad_full_sequence(data, names=("F", LENGTH), ids=ids).align_to("B", "F", "L")
    data = data.unpad_sequence()
    data = {_id: s.split(chunk_size, dim=1) for _id, s in zip(ids, data)}

    def num_remaining_chunks(data):
        return sum([len(x) for x in data.values()])

    batches = []
    while num_remaining_chunks(data) > 0:
        batch = [s.pop(0) for _id, s in data.items() if len(s) > 0]
        if len(batch) > 0:
            batches.append(pad_stream_tensor(batch).align_to(BATCH, "F", LENGTH))

    return batches


# @pytest.fixture
# def batches_of_waveform_chunks(waveforms, ids):
#     """Batches of chunks of waveforms of varying lengths from the `waveforms` data."""
#     return create_random_batches(
#         waveforms,
#         ids,
#         min_size=SAMPLE_RATE * WAVEFORM_CHUNK_MIN_SECONDS,
#         max_size=SAMPLE_RATE * WAVEFORM_CHUNK_MAX_SECONDS,
#     )


# @pytest.fixture
# def batches_of_token_chunks(tokens, ids):
#     """Batches of chunks of tokens of varying lengths from the `tokens` data."""
#     return create_random_batches(tokens, ids, min_size=TOKENS_CHUNK_MIN_SIZE, max_size=TOKENS_CHUNK_MAX_SIZE)
