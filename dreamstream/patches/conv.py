
import types
from copy import deepcopy

import torch
import torch.nn.functional as F

from dreamstream.tensor import StreamTensor, StreamMetadata
from dreamstream.patches.general import online, offline
from dreamstream.nn.utils import pad_stream_tensor

#TODO: Add support for subsampling convolutions (i.e., stride > kernel_width).

def conv_1d_pre_hook(self, inputs):
    
    if not self.streaming:
        assert not isinstance(inputs[0], StreamTensor), "Using StreamTensors in offline mode might result in unexpected behavior."
        return inputs
    
    input = inputs[0]
    
    assert isinstance(input, StreamTensor), "The input is expected to be StreamTensor when in online mode."
        
    # If all inputs are NOT first, collect states for all.
    if (self.kernel_width > 1) and (not input.meta.all_starting):
        buffer_data = [self.stream_buffer[_id] if _id in self.stream_buffer else None for _id in input.meta.ids]
        buffer_lengths = torch.as_tensor([0 if x is None else x.size(-1) for x in buffer_data])
        assert buffer_lengths.min() >= 0, "At least one buffer should have length greater than zero."
        ref_length = buffer_lengths[0]
        
        # If all have the same length, stack them and concatenate with input.
        if (buffer_lengths == ref_length).all():
            buffer_data = torch.stack(buffer_data) 
            input = torch.cat([buffer_data, input], dim=-1)
        
        # If not, split batch into individual inputs and concatenate separately first.
        else:
            # TODO: This needs to be tested.
            input = input.unpad_sequence()
            input = [a if b is None else torch.cat([b, a], dim=-1) for a, b in zip(input, buffer_data)]
            input = pad_stream_tensor(input).permute(1, 2, 0)

    return input


def conv_1d_post_hook(self, inputs, outputs):
    
    if self.streaming:
        outputs, buffer = outputs
        self.stream_buffer.update(buffer)
        
        # TODO: Simplify this.
        if outputs.meta.any_end:
            for _id, eos in zip(outputs.meta.ids, outputs.meta.eos):
                if eos and _id in self.stream_buffer:
                    del self.stream_buffer[_id]
    
    return outputs

def patch_conv_1d(module):
    
    # Add streaming mode.
    module.online = types.MethodType(online, module)
    module.offline = types.MethodType(offline, module)
    module.streaming = False
    
    # Add stream_buffer dictionary.
    module.stream_buffer = {}
    
    # Add module-specific attributes.
    module.kernel_width = module.kernel_size[0] + (module.kernel_size[0] - 1) * (module.dilation[0] - 1)
    
    # Register pre_hook and post_hook.
    module.register_forward_pre_hook(conv_1d_pre_hook)
    module.register_forward_hook(conv_1d_post_hook)
    
    return module








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
        ss1 = StreamMetadata(ids=["test_1"], lengths=[x1.size(-1)], sos=[n == 0], eos=[n == N - 1])
        #ss2 = StreamMetadata(ids=["test_1", "test_2"], lengths=[x1.size(-1), x2.size(-1)], first=[n == 0] * 2, last=[n == N - 1] * 2)
        xs1 = StreamTensor(x1, meta=ss1)
        #xs2 = StreamTensor(x2, meta=ss2)
        stream_inputs.append(xs1)
        #stream_batch_inputs.append(xs2)
        inputs.append(x1)
    
    x_full = torch.cat(inputs, dim=-1)
    s = StreamMetadata(ids=["test_2"], lengths=[x_full.size(-1)], sos=[True], eos=[True])
    x_full_stream = StreamTensor(x_full, meta=s)
    
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
    assert torch.allclose(y1, y3, rtol=1e-6, atol=1e-6)