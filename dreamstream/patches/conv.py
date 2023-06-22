import torch

from dreamstream.tensor import StreamTensor
from dreamstream.patches.modes import add_streaming_modes
from dreamstream.nn.utils import pad_stream_tensor


# TODO (LB): #1 Add support for subsampling convolutions (i.e., stride > kernel_width).
# TODO (LB): #2 Generalize to N-D convolutions, where the length dimension can be any dimension.
# TODO (LB): Add support for transposed convolutions.


def conv_1d_pre_hook(self, inputs):
    is_stream_tensor = isinstance(inputs[0], StreamTensor)
    if not self.streaming:
        if is_stream_tensor:
            raise RuntimeError("Using StreamTensors in offline mode might result in unexpected behavior.")
        return inputs

    input = inputs[0]
    if not is_stream_tensor:
        raise RuntimeError("The input is expected to be StreamTensor when in online mode.")

    # If all inputs are NOT first, collect states for all.
    if (self.kernel_width > 1) and (not input.meta.all_starting):
        buffer_data = [self.stream_buffer[_id] if _id in self.stream_buffer else None for _id in input.meta.ids]
        buffer_lengths = torch.as_tensor([0 if x is None else x.size(-1) for x in buffer_data])
        assert buffer_lengths.min() >= 0, "At least one buffer should have length greater than zero."
        ref_length = buffer_lengths[0]

        if (buffer_lengths == ref_length).all():
            # If all have the same length, stack them and concatenate with input.
            buffer_data = torch.stack(buffer_data)
            input = torch.cat([buffer_data, input], dim=-1)
        else:
            # If not, split batch into individual inputs and concatenate separately first.
            # TODO (LB): This needs to be tested.
            input = input.unpad_sequence()
            input = [x if b is None else torch.cat([b, x], dim=-1) for x, b in zip(input, buffer_data)]
            input = pad_stream_tensor(input).permute(1, 2, 0)

    return input


def conv_1d_post_hook(self, inputs, outputs):
    if self.streaming:
        outputs = outputs
        self.stream_buffer.update(outputs.meta._temp_buffer)
        outputs.meta._temp_buffer = None

        # TODO (LB): Simplify this.
        if outputs.meta.any_ending:
            for _id, eos in zip(outputs.meta.ids, outputs.meta.eos):
                if eos and _id in self.stream_buffer:
                    del self.stream_buffer[_id]

    return outputs


def patch_conv_1d(module):
    add_streaming_modes(module)

    # Add stream_buffer dictionary.
    module.stream_buffer = {}

    # Add module-specific attributes.
    module.kernel_width = module.kernel_size[0] + (module.kernel_size[0] - 1) * (module.dilation[0] - 1)

    # Register pre_hook and post_hook.
    module.register_forward_pre_hook(conv_1d_pre_hook)
    module.register_forward_hook(conv_1d_post_hook)

    return module
