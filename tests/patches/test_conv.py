import torch
import torch.nn as nn
import pytest
import torchaudio

from dreamstream import patch
from dreamstream.data.data_objects import OutputCollector
from tests.conftest import (
    WAVEFORM_CHUNK_SIZE,
    WAVEFORM_CHUNK_MIN_SIZE,
    WAVEFORM_CHUNK_MAX_SIZE,
    WAVEFORM_DIM,
    create_structured_batches,
    create_random_batches,
)


class TestConvs:
    test_modules = [
        nn.Conv1d(WAVEFORM_DIM, 4, kernel_size=1, padding=0),
        nn.Conv1d(WAVEFORM_DIM, 4, kernel_size=1, padding=1),
        nn.Conv1d(WAVEFORM_DIM, 4, kernel_size=3, padding=0),
        nn.Conv1d(WAVEFORM_DIM, 4, kernel_size=3, padding=1),
        nn.Conv1d(WAVEFORM_DIM, 4, kernel_size=4, padding=0),
        nn.Conv1d(WAVEFORM_DIM, 4, kernel_size=4, padding=2),
        nn.Conv1d(WAVEFORM_DIM, 4, kernel_size=5, padding=0),
        nn.Conv1d(WAVEFORM_DIM, 4, kernel_size=5, padding=2),
        nn.Conv1d(WAVEFORM_DIM, 4, kernel_size=5, padding=2, stride=2),
        nn.Conv1d(WAVEFORM_DIM, 4, kernel_size=5, padding=2, stride=2),
        nn.Sequential(
            nn.Conv1d(WAVEFORM_DIM, 4, kernel_size=3, padding=1),
            nn.Conv1d(4, 4, kernel_size=5, padding=2),
        ),
        nn.Sequential(
            nn.Conv1d(WAVEFORM_DIM, 4, kernel_size=3, padding=1, stride=2),
            nn.Conv1d(4, 4, kernel_size=5, padding=2, stride=2),
        ),
        torchaudio.models.Wav2Letter(
            num_classes=40, input_type="mfcc", num_features=WAVEFORM_DIM
        ),  # Too slow for remote CI.
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
    @pytest.mark.parametrize("is_structured_batches", [True, False])
    def test_equivalence(self, waveforms, ids, module, is_structured_batches):
        if is_structured_batches:
            batches = create_structured_batches(waveforms, ids, chunk_size=WAVEFORM_CHUNK_SIZE)
        else:
            batches = create_random_batches(waveforms, ids, min_size=WAVEFORM_CHUNK_MIN_SIZE, max_size=WAVEFORM_CHUNK_MAX_SIZE)

        with torch.inference_mode():
            # Offline targets
            module.offline()
            targets = {_id: module(s.unsqueeze(0)) for _id, s in zip(ids, waveforms)}

            # Online outputs
            stream_output = OutputCollector()
            module.online()
            for x in batches:
                y = module(x)
                stream_output.update(y)

        for _id, _y in targets.items():
            y = stream_output[_id].tensor()
            assert torch.allclose(_y, y, atol=1e-6), f"{_id} failed"
