from typing import Any, List, Optional, Tuple, Union
import torch

from dreamstream.tensor import StreamTensor
from dreamstream.patches.modes import add_streaming_modes
from dreamstream.nn.utils import pad_stream_tensor


# TODO (JDH): How do we deal with PackedSequence?
# TODO (JDH): When an initial state is provided, we use it only for the first chunk.)
# TODO (JDH): Deal with non-batched inputs and hidden states (note that when input is batched so must be the hidden state).


def get_tensor_and_state(
    inputs: Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if len(inputs) == 2:
        return inputs

    return inputs[0], None


class StateStore(dict):
    """A dictionary that stores states for each batch id but also supports retrieving states for individual examples."""

    def __init__(self, *args, batch_dim: int, **kwargs):
        self.batch_dim = batch_dim
        super().__init__(*args, **kwargs)
        self.batch_keys = set(k for k in self.keys() if isinstance(k, tuple))

    def __contains__(self, keys: Union[str, Tuple[str]]):
        # Single key and batch key matching.
        if keys in self.__dict__:
            return True
        
        # Find individual key matches allowing some to be part of batch keys.
        # Return True if every key is found.
        batch_key = tuple(keys)
        found_keys = 0
        for k in batch_key:
            for bk in self.batch_keys:
                if k in bk:
                    found_keys += 1

        return found_keys == len(batch_key)

    def __getitem__(self, keys: Union[str, Tuple[str]]) -> torch.Tensor:
        # Single key and batch key matching.
        try:
            return super().__getitem__(keys)
        except KeyError:
            pass

        # Find individual key matches allowing some to be part of batch keys. Returns KeyError if any key is not found.
        batch_key = tuple(keys)
        vals = []
        for k in batch_key:
            if k in self.__dict__:
                vals.append(super().__getitem__(k))
            else:
                for bk in self.batch_keys:
                    if k in bk:
                        tensor = super().__getitem__(bk)
                        sample = torch.select(tensor, dim=self.batch_dim, index=bk.index(k))
                        # if isinstance(tensor, StreamTensor) and tensor.names[dim] != BATCH:
                        #     sample = sample.align_to(*tensor.names)
                        vals.append(sample)

        if len(vals) != len(batch_key):
            found_keys = tuple(k for k in batch_key if k in self.__dict__ or any(k in bk for bk in self.batch_keys))
            missing_keys = tuple(k for k in batch_key if k not in found_keys)
            msg = f"Could not find keys {missing_keys}."
            if found_keys:
                msg += f" Found keys {found_keys}."
            raise KeyError(msg)

        return torch.stack(vals, dim=self.batch_dim).rename(*tensor.names)

    def __setitem__(self, key: Union[str, Tuple[str]], value: torch.Tensor):
        super().__setitem__(key, value)
        if isinstance(key, tuple):
            self.batch_keys.add(key)

    def __delitem__(self, keys):
        import IPython
        IPython.embed(using=False, header="StateStore.__delitem__")
        # Single key and batch key matching.
        if keys in self.__dict__:
            del self.__dict__[keys]
            return None
        
        # Find individual key matches allowing some to be part of batch keys.
        # Return True if every key is found.
        batch_key = tuple(keys)
        found_keys = 0
        for k in batch_key:
            for bk in self.batch_keys:
                if k in bk:
                    found_keys += 1

        return found_keys == len(batch_key)


def rnn_pre_hook(self, inputs):
    """
    
    Cases:
    - Given state:
      1. None
      2. Tensor
    - Given ids:
      1. All are first
      2. Some are first
      3. None are first
      4. All are last
      5. Some are last
      6. None are last
      7. All are first and last
      8. Some are first and some are last
      9. None are first or last
    - Hidden state store:
      1. All ids are in the hidden state store
      2. Some ids are in the hidden state store
      3. No ids are in the hidden state store

    Get hidden state (num_layers, batch_size, hidden_size).
    If any of the ids are in the hidden state store, we must use that state (unless they are the first chunk, but they can't be if they are in the hidden state store).
    Any first chunks must use the given initial state.
      1. If all chunks are first chunks, we can use the given initial state whether it is None or Tensor.
      2. If only some chunks are first chunks, we must use the given initial state for those chunks and the hidden
         state store for the others by writing selectively.

    Args:
        self (nn.RNN): The current module.
        inputs (Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]): Inputs to `self.forward`.

    Raises:
        RuntimeError: If input is a StreamTensor and the module is not in online mode, or if the input is not a 
            StreamTensor but the module is in online mode.

    Returns:
        (Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]): Modified inputs to `self.forward`.
    """
    x, state = get_tensor_and_state(inputs)

    input_is_stream_tensor = isinstance(x, StreamTensor)
    if not self.streaming:
        state_is_stream_tensor = isinstance(state, StreamTensor) if state is not None else False
        if input_is_stream_tensor or state_is_stream_tensor:
            raise RuntimeError("Using StreamTensors in offline mode might result in unexpected behavior.")
        return inputs

    if not input_is_stream_tensor:
        raise RuntimeError("The input is expected to be StreamTensor when in online mode.")

    # import IPython
    # IPython.embed(using=False)

    # TODO (JDH): Deal with initial hidden state here.
    ids_without_state = [(i, _id) for i, _id in enumerate(x.meta.ids) if _id not in self.hidden_state_store]
    if any(ids_without_state):
        if len(ids_without_state) == len(x.meta.ids) and state is None:
            # No ids have state and no custom state is given.
            state = None
        elif len(ids_without_state) == len(x.meta.ids) and state is not None:
            # No ids have state but a custom state is given.
            default_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            state = default_state
        elif len(ids_without_state) < len(x.meta.ids) and state is None:
            # Some ids have state but no custom state is given.
            for i, _id in ids_without_state:
                self.hidden_state_store[_id] = torch.zeros(self.num_layers, 1, self.hidden_size, device=x.device)
            state = self.hidden_state_store[x.meta.ids]
        else:
            # Some ids have state but a custom state is given too.
            # TODO (JDH): Speed this up by using torch.select_scatter or similar directly on the given state.
            for i, _id in ids_without_state:
                self.hidden_state_store[_id] = state[:, i, :].unsqueeze(1)
            state = self.hidden_state_store[x.meta.ids]
            # # Write default_state into the state tensor at the batch ids that do not have state.
            # index = torch.as_tensor(ids_without_state, device=x.device)
            # dim = 0 if self.batch_first else 1
            # torch.select_scatter(state, default_state, dim=dim, index=index)
    else:
        # All ids have state. Return the state.
        state = self.hidden_state_store[x.meta.ids]

    x.meta._temp_names = x.names
    x = x.rename(None)
    if state is not None:
        state = state.rename(None)
    return x, state


def rnn_post_hook(self, inputs, outputs):
    if not self.streaming:
        return outputs

    input, in_state = get_tensor_and_state(inputs)
    output, out_state = get_tensor_and_state(outputs)

    # Ensure out_state is a proper StreamTensor.
    # TODO (JDH): Not sure if we want to propagate the StreamTensor onto states
    # out_state = StreamTensor(out_state, meta=input.meta.clone())
    # out_state.meta.lengths = torch.ones_like(out_state.meta.lengths)
    # out_state.rename_(*("num_layers", input.meta._temp_names[0], input.meta._temp_names[-1]))
    if isinstance(out_state, StreamTensor):
        out_state = out_state.tensor()

    # Store hidden states for this batch
    self.hidden_state_store[input.meta.ids] = out_state

    # Delete hidden states of ending files.
    ending_ids = tuple(_id for _id, eos in zip(input.meta.ids, input.meta.eos) if eos)
    if ending_ids:
        del self.hidden_state_store[ending_ids]
   
    # Ensure output is a proper StreamTensor.
    output.rename_(*input.meta._temp_names)
    return StreamTensor(output, meta=input.meta), out_state


def patch_rnn(module):
    add_streaming_modes(module)

    # Add a dictionary for storing previous hidden states of shape (D * num_layers, N, H).
    module.hidden_state_store = StateStore(batch_dim=1)

    # Register pre_hook and post_hook.
    module.register_forward_pre_hook(rnn_pre_hook)
    module.register_forward_hook(rnn_post_hook)
    return module
