import random
import torch
import torch.nn as nn
import pytest

from dreamstream import patch
from dreamstream.data.data_objects import OutputCollector
from dreamstream.utils.flags import BATCH, LENGTH


class TestRNNs:
    initial_h = torch.randn(1, 1, 16)
    initial_c = torch.randn(1, 1, 16)

    test_modules = [
        nn.RNN(1, 16, bidirectional=False, batch_first=True),
        # nn.Sequential(
        #     nn.RNN(1, 4, bidirectional=False, batch_first=True),
        #     nn.RNN(4, 4, bidirectional=False, batch_first=True),
        # ),
        # nn.GRU(1, 16, bidirectional=False, batch_first=True),
        # nn.Sequential(
        #     nn.GRU(1, 4, bidirectional=False, batch_first=True),
        #     nn.GRU(4, 4, bidirectional=False, batch_first=True),
        # ),
    ]

    def recursive_assert(self, module):
        assert hasattr(module, "online")
        assert hasattr(module, "offline")

        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            assert hasattr(module, "stream_buffer")
            assert hasattr(module, "kernel_width")

        if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            pass

    @pytest.mark.parametrize("module", test_modules)
    def test_patch(self, module):
        patch(module)
        module.apply(self.recursive_assert)

    @pytest.mark.parametrize("initial_state", [False])
    @pytest.mark.parametrize("module", test_modules)
    def test_equivalence(self, waveforms, ids, batches_of_waveform_chunks, module, initial_state):
        if initial_state:
            if isinstance(module, nn.LSTM):
                state = (self.initial_h, self.initial_c)
            else:
                state = self.initial_h
        else:
            state = None

        with torch.inference_mode():
            targets = {_id: module(s.transpose(0, 1).unsqueeze(0), state) for _id, s in zip(ids, waveforms)}

        # TODO (JDH): Test final state equivalence
        stream_output = OutputCollector()
        module.online()
        with torch.inference_mode():
            for x in batches_of_waveform_chunks:
                y, state = module(x.align_to(BATCH, LENGTH, "F"), state)
                stream_output.update(y)

        import IPython
        IPython.embed(using=False)
        for _id, _y in targets.items():
            y = stream_output[_id].tensor()
            assert torch.allclose(_y, y, atol=1e-6), f"{_id} failed"
