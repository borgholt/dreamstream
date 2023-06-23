import torch
import torch.nn as nn
import pytest
import torchaudio

from dreamstream import patch
from dreamstream.data.data_objects import OutputCollector


class TestConvs:
    test_modules = [
        nn.Conv1d(16, 16, kernel_size=1, padding=0),
        nn.Conv1d(16, 16, kernel_size=1, padding=1),
        nn.Conv1d(16, 16, kernel_size=3, padding=0),
        nn.Conv1d(16, 16, kernel_size=3, padding=1),
        nn.Conv1d(16, 16, kernel_size=4, padding=0),
        nn.Conv1d(16, 16, kernel_size=4, padding=2),
        nn.Conv1d(16, 16, kernel_size=5, padding=0),
        nn.Conv1d(16, 16, kernel_size=5, padding=2),
        nn.Conv1d(16, 16, kernel_size=5, padding=2, stride=2),
        nn.Conv1d(16, 16, kernel_size=5, padding=2, stride=2),
        nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
        ),
        nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1, stride=2),
            nn.Conv1d(16, 16, kernel_size=5, padding=2, stride=2),
        ),
        # torchaudio.models.Wav2Letter(num_classes=40, input_type="mfcc", num_features=16),  # Too slow for remote CI.
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
