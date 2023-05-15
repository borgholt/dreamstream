import types
import uuid
from collections import defaultdict
from copy import deepcopy

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# Potential data preparation functions:
# - C chunks from input
# - get first state of size N


class StreamTensor(torch.Tensor):
    
    @staticmethod 
    def __new__(cls, x, stream_state, *args, **kwargs): 
        return super().__new__(cls, x, *args, **kwargs) 
    
    def __init__(self, x, stream_state, *args, **kwargs):
        self._stream_state = stream_state
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        

        stream_state = [x._stream_state for x in args if isinstance(x, StreamTensor)][0]
        # TODO: If more than one, assert that stream_states are identical, or raise error
        out = super().__torch_function__(func, types, args, kwargs)
        
        if isinstance(out, torch.Tensor):
            return StreamTensor(out, stream_state=stream_state)
        else:
            return out
    
    @property
    def stream_state(self):
        return self._stream_state
        
    @stream_state.setter
    def stream_state(self, s):
        self._stream_state = s
        
    
def stream_tensor(data, stream_state, *args, dtype=None, device=None, requires_grad=False, pin_memory=False):
    x = torch.tensor(data, *args, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)
    return StreamTensor(x, stream_state)

def lengths_from_list(data, time_dim=-1):
    return torch.tensor([x.shape[time_dim] for x in data])

        
class StreamState():
    
    def __init__(self, ids, lengths, first, last) -> None:
        
        self.ids = ids
        self.lengths = torch.tensor(lengths)
        self.first = torch.tensor(first)
        self.last = torch.tensor(last)
        self.empty = torch.full((len(ids),), False)
        
        assert (self.lengths > 0).any(), "lengths must be all positive"

        # TODO: Consider if this is worthwhile. Must be updated when StreamTensors are concatenated, stacked, etc. 
        self._any_first = self.first.any().item()
        self._any_last = self.last.any().item()
        
        self._all_first = self.first.all().item()
        self._all_last = self.last.all().item()
        
        self._any_first_or_last = self._any_first or self._any_last
        
    def __repr__(self):
        return f"StreamState(size={len(self.ids)})"
    
    @property
    def _first_lengths(self):
        return self.lengths[self.first]
    
    @property
    def _last_lengths(self):
        return self.lengths[self.last]
    
    @_first_lengths.setter
    def _first_lengths(self, i):
        self.lengths[self.first] = i
    
    @_last_lengths.setter
    def _last_lengths(self, i):
        self.lengths[self.last] = i

                

def set_padding_to_zero(x, sl):
    # TODO: Implement
    return x



def conv_1d_streaming_forward(self, input):
    
    if not self.streaming:
        return self.original_forward(input)
    
    assert isinstance(input, StreamTensor), "input is expected to be StreamTensor when in streaming mode"
    
    # TODO: Check if input is None (or another object that indicates that computation can be skipped)
    
    # Create shortform variables for padding (p), kernel size (k), and stride (s).
    p = self.original_padding[0] if isinstance(self.original_padding, tuple) else 0
    k = self.kernel_size[0]
    k += (k - 1) * (self.dilation[0] - 1)
    s = self.stride[0]
    
    # TODO: Check that input is greater than kernel size, otherwise PyTorch will throw an error. Maybe this should be done after padding and adding states.
    
    stream_state = input.stream_state # DELETE when "StreamTensor" functions are implemented.
    input = torch.Tensor(input) # DELETE when "StreamTensor" functions are implemented.
    
    # If all inputs are NOT first, collect states for all.
    if (k > 1) and (not stream_state._all_first):
        state_data = [self.input_states[_id] if _id in self.input_states else None for _id in stream_state.ids]
        state_lengths = torch.as_tensor([0 if x is None else x.size(-1) for x in state_data])
        assert state_lengths.min() >= 0, "some state lengths should be greater than zero"
        ref_length = state_lengths[0]
        # If all have the same length, stack them and concatenate with input.
        if (state_lengths == ref_length).all():
            state_data = torch.stack(state_data)
            input = torch.cat([state_data, input], dim=-1)
            stream_state.lengths += ref_length
        # If not, split batch into individual inputs and concatenate separately first.
        else:
            input = torch.nn.utils.rnn.unpad_sequence(input.permute(2, 0, 1), stream_state.lengths)
            input = [x if y is None else torch.cat([y, x]) for x, y in zip(input, state_data)]
            stream_state.lengths += state_lengths
            input = torch.nn.utils.rnn.pad_sequence(input).permute(1, 2, 0)
            
    # Update sequence lengths to include padding, which is used to compute output lengths.
    if stream_state._any_first_or_last and p != 0:
        stream_state._first_lengths += p
        stream_state._last_lengths += p
        
        # Add padding to input if necessary.
        total_p = stream_state.lengths.max().item() - input.size(-1)
        if stream_state._all_first:
            assert total_p >= p, "total padding should be greater than or equal to padding"
            input = F.pad(input, (p, total_p - p))
        elif total_p > 0:
            input = F.pad(input, (0, total_p))
            if stream_state._any_first:
                input[stream_state.first] = torch.roll(input[stream_state.first], shifts=p, dims=-1)
                
    # Compute output lengths and updates states.
    output_lengths = ((stream_state.lengths - k) // s + 1).clip(min=0)
    next_start_pos = output_lengths * s
    if k > 1:
        for i, (start, end, _id) in enumerate(zip(next_start_pos, stream_state.lengths, stream_state.ids)):
            self.input_states[_id] = input[i, ..., start:end]
    
    # Convolve input. 
    stream_state.lengths = output_lengths
    if stream_state.lengths.max().item() == 0:
        input = None
    else:
        input = self.original_forward(input)
        input = StreamTensor(input, stream_state)
    
    return input
    
def online(self, mode=True):
    if not isinstance(mode, bool):
        raise ValueError("streaming mode is expected to be boolean")
    self.streaming = mode
    self._module_specific_streaming(mode)
    for module in self.children():
        module.online(mode)
    if isinstance(self, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        self.padding = (0,) if mode else self.original_padding
    return self

def offline(self):
    return self.online(mode=False)
    
def add_bound_method_state_template_(module, template):
    module._state_template = types.MethodType(template, module)

def conv_1d_state_template(self, data):
    return dict(data=None)

def conv_1d_module_specific_streaming(self, mode):
    if mode:
        self.padding = (0,)
    else:
        self.padding = self.original_padding

def patch_conv_1d(module):
    
    # add streaming mode
    module._module_specific_streaming = types.MethodType(conv_1d_module_specific_streaming, module)
    module.online = types.MethodType(online, module)
    module.offline = types.MethodType(offline, module)
    module.streaming = False
    
    # add input_states
    module._state_template = types.MethodType(conv_1d_state_template, module)
    module.input_states = {}
    
    # verify padding behavior
    assert module.padding_mode == "zeros", "padding mode is expected to be 'zeros'"
    assert module.padding != "same", "'same' padding not supported for streaming"
    module.original_padding = deepcopy(module.padding)
    
    # adjust forward behavior
    module.original_forward = module.forward
    module.forward = types.MethodType(conv_1d_streaming_forward, module)
    
    return module

    
    
    
    
    
    

# def _replace_tuple_element(x, index, value):
#     return tuple(value if i == index else v for i, v in enumerate(x))

    

if __name__ == '__main__':
    
    import os
    import argparse
    from random import randint
    from torch.nn.utils.rnn import pad_sequence
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    parser = argparse.ArgumentParser()

    # configuration of layer
    parser.add_argument("--in_channels", default=32, type=int)
    parser.add_argument("--out_channels", default=64, type=int)
    parser.add_argument("--kernel_size", default=3, type=int)
    parser.add_argument("--stride", default=2, type=int)

    # configuration of test input
    parser.add_argument("--input_min_length", default=1000, type=int)
    parser.add_argument("--input_max_length", default=2000, type=int)
    parser.add_argument("--input_min_chunk_size", default=50, type=int)
    parser.add_argument("--input_max_chunk_size", default=50, type=int)
    parser.add_argument("--input_batch_size", default=32, type=int)
    
    args, _ = parser.parse_known_args()

    assert args.input_min_length <= args.input_max_length
    assert args.input_min_chunk_size <= args.input_max_chunk_size
    
    N = 20_000
    stream_inputs, stream_batch_inputs, inputs = [], [], []
    for n in range(N):
        x1 = torch.rand(1, 32, randint(1, 20))
        #x2 = pad_sequence([x1, torch.rand(1, 32, randint(args.kernel_size, 50))], batch_first=True, padding_value=0)
        ss1 = StreamState(ids=["test_1"], lengths=[x1.size(-1)], first=[n == 0], last=[n == N - 1])
        #ss2 = StreamState(ids=["test_1", "test_2"], lengths=[x1.size(-1), x2.size(-1)], first=[n == 0] * 2, last=[n == N - 1] * 2)
        xs1 = StreamTensor(x1, stream_state=ss1)
        #xs2 = StreamTensor(x2, stream_state=ss2)
        stream_inputs.append(xs1)
        #stream_batch_inputs.append(xs2)
        inputs.append(x1)
    
    x_full = torch.cat(inputs, dim=-1)
    s = StreamState(ids=["test_2"], lengths=[x_full.size(-1)], first=[True], last=[True])
    x_full_stream = StreamTensor(x_full, stream_state=ss_full)
    
    conv1d = torch.nn.Conv1d(args.in_channels, args.out_channels, args.kernel_size, args.stride, padding=2)
    
    y1 = conv1d(x_full)
    
    conv1d = patch_conv_1d(conv1d)
    conv1d.online()
    
    y2 = conv1d(x_full_stream)
    
    ys = []
    for x in stream_inputs:
        y = conv1d(x)
        if y is None:
            print("skipping")
            continue
        ys.append(y)
    y3 = torch.cat([torch.Tensor(y_) for y_ in ys], dim=-1)
    
    assert torch.allclose(y1, y2, rtol=0, atol=0)
    assert torch.allclose(y1, y3, rtol=0, atol=0)
    
    



