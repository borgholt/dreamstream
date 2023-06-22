from typing import Optional, Tuple, Union
import torch

from dreamstream.tensor import StreamTensor
from dreamstream.patches.modes import add_streaming_modes
from dreamstream.nn.utils import pad_stream_tensor


# TODO (JDH): How do we deal with PackedSequence?
# TODO (JDH): When an initial state is provided, we use it only for the first chunk.)


def get_tensor_and_state(inputs: Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if len(inputs) == 2:
        return inputs

    return inputs[0], None


def rnn_pre_hook(self, inputs):
    # TODO (JDH): Deal with initial hidden state here.
    x, state = get_tensor_and_state(inputs)

    is_stream_tensor = isinstance(x, StreamTensor)
    if not self.streaming:
        if is_stream_tensor:
            raise RuntimeError("Using StreamTensors in offline mode might result in unexpected behavior.")
        return inputs

    if not is_stream_tensor:
        raise RuntimeError("The input is expected to be StreamTensor when in online mode.")
    
    # Cases:
    # - Given state:
    #   1. None
    #   2. Tensor
    # - Given ids:
    #   1. All are first
    #   2. Some are first
    #   3. None are first
    #   4. All are last
    #   5. Some are last
    #   6. None are last
    #   7. All are first and last
    #   8. Some are first and some are last
    #   9. None are first or last
    # - Hidden state store:
    #   1. All ids are in the hidden state store
    #   2. Some ids are in the hidden state store
    #   3. No ids are in the hidden state store

    # Get hidden state (num_layers, batch_size, hidden_size).
    # If any of the ids are in the hidden state store, we must use that state (unless they are the first chunk, but they can't be if they are in the hidden state store).
    # Any first chunks must use the given initial state.
    #   1. If all chunks are first chunks, we can use the given initial state whether it is None or Tensor.
    #   2. If only some chunks are first chunks, we must use the given initial state for those chunks and the hidden 
    #      state store for the others by writing selectively.
    
    missing_ids = [i for i, _id in enumerate(x.meta.ids) if _id in self.hidden_state_store]
    if any(missing_ids):
        default_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        if state is not None:
            # Write default_state into the state tensor at the batch ids where ~is_id_present.
            index = torch.as_tensor(missing_ids, device=x.device)
            dim = 0 if self.batch_first else 1
            torch.select_scatter(state, default_state, dim=dim, index=index)
        
        state = torch.stack([self._hidden_state_store[_id] for _id in x.meta.ids], dim=1)
    else:
        for i, _id in enumerate(x.meta.ids):
            if _id in self.hidden_state_store:
                state.append(self.hidden_state_store[_id])
        default_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        

    # import IPython
    # IPython.embed(using=False)
    
    x.meta._temp_names = x.names
    x = x.rename(None)
    return x, state


def rnn_post_hook(self, inputs, outputs):
    if not self.streaming:
        return outputs
    
    # import IPython
    # IPython.embed(using=False)

    input, in_state = get_tensor_and_state(inputs)
    output, out_state = get_tensor_and_state(outputs)

    # Store hidden state
    # TODO (JDH): Default to storing the batched hidden state.
    for i, _id in enumerate(output.meta.ids):
        self.hidden_state_store[_id] = out_state[:, i, :]

    output.rename_(*input.meta._temp_names)

    if isinstance(out_state, StreamTensor):
        out_state.rename_(*input.meta._temp_names)

    return StreamTensor(output, meta=input.meta), out_state


def patch_rnn(module):
    add_streaming_modes(module)

    # Add a dictionary for storing previous hidden states.
    module.hidden_state_store = {}

    # Register pre_hook and post_hook.
    module.register_forward_pre_hook(rnn_pre_hook)
    module.register_forward_hook(rnn_post_hook)
    return module

