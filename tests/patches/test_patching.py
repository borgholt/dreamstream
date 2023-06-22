import random
import torch
import torch.nn as nn
import torchaudio
import pytest

from dreamstream import patch
from dreamstream.data.data_objects import OutputCollector
from dreamstream.nn.utils.pad_sequence import pad_full_sequence, pad_stream_tensor
from dreamstream.utils.flags import LENGTH


torch.manual_seed(42)
random.seed(42)


def random_chunks(full_length, min_size: int = 1000, max_size: int = 8000):
    chunks = []
    chunk_sum, remaining = 0, full_length
    while remaining > 0:
        chunks.append(min(random.randint(min_size, max_size), remaining))
        chunk_sum = sum(chunks)
        remaining = full_length - chunk_sum
    return chunks


BATCH_SIZE = 32


@pytest.fixture
def sequences():
    return [torch.rand(16, random.randint(16000 * 5, 16000 * 8)) for i in range(BATCH_SIZE)]


@pytest.fixture
def ids():
    return [str(hash(i)) for i in range(BATCH_SIZE)]


@pytest.fixture
def batches_of_chunks(sequences, ids):
    data = pad_full_sequence(sequences, names=("F", LENGTH), ids=ids).align_to("B", "F", "L")
    data = data.unpad_sequence()
    data = {
        _id: s.split(random_chunks(s.size("L"), min_size=16000, max_size=32000), dim=1) for _id, s in zip(ids, data)
    }

    def num_remaining_chunks(data):
        return sum([len(x) for x in data.values()])

    batches = []
    while num_remaining_chunks(data) > 0:
        batch = [s.pop(0) for _id, s in data.items() if len(s) > 0 and random.random() < 0.75]
        if len(batch) > 0:
            batches.append(pad_stream_tensor(batch).align_to("B", "F", "L"))

    return batches


class TestPatching:
    test_modules = [
        nn.Conv1d(16, 128, kernel_size=1, padding=0),
        nn.Conv1d(16, 128, kernel_size=1, padding=1),
        nn.Conv1d(16, 128, kernel_size=3, padding=0),
        nn.Conv1d(16, 128, kernel_size=3, padding=1),
        nn.Conv1d(16, 128, kernel_size=4, padding=0),
        nn.Conv1d(16, 128, kernel_size=4, padding=2),
        nn.Conv1d(16, 128, kernel_size=5, padding=0),
        nn.Conv1d(16, 128, kernel_size=5, padding=2),
        nn.Conv1d(16, 128, kernel_size=5, padding=2, stride=2),
        nn.Conv1d(16, 128, kernel_size=5, padding=2, stride=2),
        nn.Sequential(
            nn.Conv1d(16, 128, kernel_size=3, padding=1),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
        ),
        nn.Sequential(
            nn.Conv1d(16, 128, kernel_size=3, padding=1, stride=2),
            nn.Conv1d(128, 128, kernel_size=5, padding=2, stride=2),
        ),
        torchaudio.models.Wav2Letter(num_classes=40, input_type="mfcc", num_features=16),
        # GreedyCTCDecoder(labels=ascii_lowercase[:14] + " " + BLANK_TOKEN, blank_index=15),
    ]

    def recursive_assert(self, module):
        assert hasattr(module, "online")
        assert hasattr(module, "offline")
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            assert hasattr(module, "stream_buffer")
            assert hasattr(module, "kernel_width")

    @pytest.mark.parametrize("module", test_modules)
    def test_patch(self, module):
        patch(module)
        module.apply(self.recursive_assert)

    @pytest.mark.parametrize("module", test_modules)
    def test_equivalence(self, sequences, ids, batches_of_chunks, module):
        with torch.inference_mode():
            targets = {_id: module(s.unsqueeze(0)) for _id, s in zip(ids, sequences)}

        stream_output = OutputCollector()
        module.online()
        with torch.inference_mode():
            for x in batches_of_chunks:
                y = module(x)
                stream_output.update(y)

        for _id, _y in targets.items():
            y = stream_output[_id].tensor()
            assert torch.allclose(_y, y, atol=1e-6), f"{_id} failed"
