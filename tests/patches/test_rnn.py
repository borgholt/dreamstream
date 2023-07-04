import collections
import random

import torch
import torch.nn as nn
import pytest

from dreamstream import patch
from dreamstream.data.data_objects import OutputCollector
from dreamstream.utils.flags import BATCH, LENGTH
from tests.conftest import (
    create_random_batches,
    create_structured_batches,
    TOKEN_CHUNK_SIZE,
    TOKEN_CHUNK_MIN_SIZE,
    TOKEN_CHUNK_MAX_SIZE,
    TOKEN_DIM,
    BATCH_SIZE
)


class TestRNNs:
    # initial_h = torch.randn(1, BATCH_SIZE, 16)
    # initial_c = torch.randn(1, BATCH_SIZE, 16)
    initial_h = torch.randn(1, 1, 4)
    initial_c = torch.randn(1, 1, 4)

    test_modules = [
        nn.RNN(TOKEN_DIM, 4, num_layers=1, bidirectional=False, batch_first=True),
        nn.RNN(TOKEN_DIM, 4, num_layers=2, bidirectional=False, batch_first=True),
        # nn.GRU(TOKEN_DIM, 16, num_layers=1, bidirectional=False, batch_first=True),
        # nn.GRU(TOKEN_DIM, 16, num_layers=2,  bidirectional=False, batch_first=True),
        # nn.LSTM(TOKEN_DIM, 16, num_layers=1, bidirectional=False, batch_first=True),
        # nn.LSTM(TOKEN_DIM, 16, num_layers=2,  bidirectional=False, batch_first=True),
    ]

    def recursive_assert(self, module):
        assert hasattr(module, "online")
        assert hasattr(module, "offline")

        if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            assert hasattr(module, "hidden_state_store")

    @pytest.mark.parametrize("module", test_modules)
    def test_patch(self, module):
        patch(module)
        module.apply(self.recursive_assert)

    # @pytest.mark.parametrize("use_initial_state", [False, True])
    # @pytest.mark.parametrize("is_packed_sequence", [False, True])
    # @pytest.mark.parametrize("is_structured_batches", [True, False])
    @pytest.mark.parametrize("use_initial_state", [False])
    @pytest.mark.parametrize("is_packed_sequence", [True])
    @pytest.mark.parametrize("is_structured_batches", [True, False])
    @pytest.mark.parametrize("module", test_modules)
    def test_equivalence(self, tokens, ids, module, use_initial_state, is_packed_sequence, is_structured_batches):
        lengths = torch.tensor([t.shape[1] for t in tokens])
        # Create batches.
        if is_structured_batches:
            batches = create_structured_batches(tokens, ids, chunk_size=TOKEN_CHUNK_SIZE)
        else:
            batches = create_random_batches(tokens, ids, min_size=TOKEN_CHUNK_MIN_SIZE, max_size=TOKEN_CHUNK_MAX_SIZE)

        # (Maybe) Create packed sequences
        tokens = [t.transpose(0, 1) for t in tokens]  # (D, L) -> (L, D)
        batches = [b.align_to(BATCH, LENGTH, "F") for b in batches]  # (B, L, F)
        if is_packed_sequence:
            tokens = [torch.nn.utils.rnn.pack_sequence([t]) for t in tokens]
            batches = [torch.nn.utils.rnn.pack_padded_sequence(b.align_to(BATCH, LENGTH, "F"), b.meta.lengths, batch_first=True, enforce_sorted=False) for b in batches]

        # (Maybe) Set initial state
        if use_initial_state:
            initial_state = (self.initial_h, self.initial_c) if isinstance(module, nn.LSTM) else self.initial_h
        else:
            initial_state = None

        # Run offline and online versions
        with torch.inference_mode():
            # Offline targets
            module.offline()
            targets = {
                _id: module(t, initial_state.squeeze(0) if use_initial_state else None)
                for _id, t in zip(ids, tokens)
            }
            if is_packed_sequence:
                targets = {
                    _id: (torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True)[0].squeeze(0), out_states)
                    for _id, (y, out_states) in targets.items()
                }

            # Online outputs
            module.online()
            stream_output = OutputCollector(collection="cat")
            # stream_states = OutputCollector(collection="append")
            stream_states = dict()
            for j, x in enumerate(batches):
                y, out_states = module(x, initial_state)
                if is_packed_sequence:
                    y = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True)[0]
                stream_output.update(y)
                for i, _id in enumerate(y.meta.ids):
                    # print(j, out_states[:, i], targets[_id][1])
                    if y.meta.eos[i]:
                        stream_states[_id] = out_states[:, i]
                # stream_states.update(out_states)

        # Compare outputs
        failed_outputs = []
        failed_states = []
        for _id, (_y, _state) in targets.items():
            y = stream_output[_id].tensor()
            state = stream_states[_id]

            # print(f"Online output: {y}")
            # print(f"Offline output: {_y}")
            print(_id)
            print(f"Online state: {state}")
            print(f"Offline state: {_state}")
            print((_y - y).abs().sum(0).max())
            print((_state - state).abs().sum(0).max())
            
            import IPython
            IPython.embed(using=False)
            
            if not torch.allclose(_y, y, atol=1e-6):
                num_errs = ((_y - y).abs().sum(-1) > 1e-6).sum()
                failed_outputs.append(f"{_id} failed on n={num_errs} of {y.shape[0]} output steps.")
            if not torch.allclose(_state, state, atol=1e-6):
                failed_states.append(f"{_id} failed on state, {_state} != {state}")

        if any(failed_states) or any(failed_outputs):
            raise AssertionError(f"Failed on:\n{failed_outputs}\n{failed_states}")
