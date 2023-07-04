import collections
import logging

from typing import Any, List, Optional, Tuple, Union

import torch

from dreamstream.tensor import StreamTensor
from dreamstream.patches.modes import add_streaming_modes
from dreamstream.nn.utils import pad_stream_tensor
from dreamstream.utils.flags import BATCH


# TODO (JDH): How do we deal with PackedSequence?
# TODO (JDH): When an initial state is provided, we use it only for the first chunk.)
# TODO (JDH): Deal with non-batched inputs and hidden states (note that when input is batched so must be the hidden state).


LOGGER = logging.getLogger(__file__)


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

        self.individual_keys = set()
        self.batch_keys = set()
        for k in self.keys():
            if isinstance(k, tuple):
                self.batch_keys.add(k)
            else:
                self.individual_keys.add(k)

    def get_batch_key(self, keys: Union[Any, Tuple[Any]]):
        return keys if isinstance(keys, tuple) else (keys,)

    def __contains__(self, keys: Union[Any, Tuple[Any]]):
        # Return quickly if there is an non-fragmented individual or batch key match.
        if keys in self.keys():  # or not isinstance(keys, tuple) and len(keys) == 1:
            return True

        # If no batch keys exist, then no fragmented matches will be found.
        if not self.batch_keys:
            return False

        # Find individual key matches allowing some to be part of batch keys.
        return self.__contains_fragmented__(keys)

    def __contains_fragmented__(self, keys: Union[Any, Tuple[Any]]):
        """Check if keys exist in the StateStore in a fragmented way (i.e. split across batched and individual keys)."""
        batch_key = self.get_batch_key(keys)
        found_ids = 0
        for k in batch_key:
            if k in self.individual_keys:
                found_ids += 1
            else:
                for bk in self.batch_keys:
                    if k in bk:
                        found_ids += 1

        return found_ids == len(batch_key)

    # def __contains_fragmented__(self, keys: Union[Any, Tuple[Any]]):
    #     """Check if keys exist in the StateStore in a fragmented way (i.e. split across batched and individual keys)."""
    #     batch_key = self.get_batch_key(keys)
    #     individual_finds = 0
    #     sub_batch_finds = 0
    #     for k in batch_key:
    #         if k in self.individual_keys:
    #             individual_finds += 1
    #         else:
    #             for bk in self.batch_keys:
    #                 if k in bk:
    #                     sub_batch_finds += 1

    #     return individual_finds > 0 and sub_batch_finds > 0 and individual_finds + sub_batch_finds == len(batch_key)

    def __getitem__(self, keys: Union[Any, Tuple[Any]]) -> torch.Tensor:
        print(f"__getitem__: {keys}")
        # Return quickly if there is an non-fragmented individual or batch key match.
        if keys in self.keys():
            print(f"Pulling direct match:\n{super().__getitem__(keys)}")
            return super().__getitem__(keys)
        # if not isinstance(keys, tuple) and len(keys) == 1:
        #     return super().__getitem__(keys[0])

        # Find individual key matches allowing some to be part of batch keys. Returns KeyError if any key is not found.
        batch_key = (keys,) if isinstance(keys, str) else keys
        vals = []
        for k in batch_key:
            if k in self.keys():
                tensor = super().__getitem__(k)
                print(f"Pulling individual:\n{tensor}")
                vals.append(tensor)
            else:
                for bk in self.batch_keys:
                    if k in bk:
                        tensor = super().__getitem__(bk)
                        print(f"Pulling batch:\n{tensor}")
                        sample = torch.select(tensor, dim=self.batch_dim, index=bk.index(k)).unsqueeze(self.batch_dim)
                        # if isinstance(tensor, StreamTensor) and tensor.names[dim] != BATCH:
                        #     sample = sample.align_to(*tensor.names)
                        vals.append(sample)

        if len(vals) != len(batch_key):
            found_keys = tuple(k for k in batch_key if k in self.keys() or any(k in bk for bk in self.batch_keys))
            missing_keys = tuple(k for k in batch_key if k not in found_keys)
            msg = f"Could not find keys {missing_keys}."
            if found_keys:
                msg += f" Found keys {found_keys}."
            raise KeyError(msg)

        return torch.cat(vals, dim=self.batch_dim).rename(*tensor.names)

    def __setitem__(self, key: Union[Any, Tuple[Any]], value: torch.Tensor):
        print(f"__setitem__: {key},\n{value}")
        if self.__contains_fragmented__(key):
            # We must delete the fragmented locations to make sure we correctly overwrite the state.
            # TODO (JDH): Implement this
            self.__deltitem_fragmented__(key)

        super().__setitem__(key, value)
        if isinstance(key, tuple):
            self.batch_keys.add(key)
        else:
            self.individual_keys.add(key)

    def __delitem__(self, keys):
        # Delete quickly if there is an non-fragmented individual or batch key match.
        print(f"__delitem__: {keys}")
        if keys in self.keys():
            super().__delitem__(keys)
            if isinstance(keys, tuple):
                self.batch_keys.remove(keys)
            else:
                self.individual_keys.remove(keys)

            return None

        self.__deltitem_fragmented__(keys)

    def __deltitem_fragmented__(self, keys: Union[Any, Tuple[Any]]):
        # print(f"{self.individual_keys=}")
        # print(f"{self.batch_keys=}")
        # print(f"{keys=}")
        print(f"__delitem_fragmented__: {keys}")
        batch_key = self.get_batch_key(keys)
        individual_keys_to_delete = set()
        keys_to_delete_in_batch_keys = collections.defaultdict(set)
        for k in batch_key:
            if k in self.individual_keys:
                individual_keys_to_delete.add(k)
            else:
                for bk in self.batch_keys:
                    if k in bk:
                        keys_to_delete_in_batch_keys[bk].add(k)

        self.individual_keys = self.individual_keys - individual_keys_to_delete
        for key_to_delete in individual_keys_to_delete:
            super().__delitem__(key_to_delete)

        for bk, keys_to_delete in keys_to_delete_in_batch_keys.items():
            self.batch_keys.remove(bk)
            if len(keys_to_delete) == len(bk):
                # Delete entire batch key
                super().__delitem__(bk)
            else:
                # Delete individual keys in the batch key also indexing the tensor.
                # We assume that since we are deleting part of batch key, then the kept subbatch keys should be simply
                # stored as individual keys.
                tensor = super().__getitem__(bk)
                # import IPython
                # IPython.embed(using=False)
                keys_to_keep = tuple(k for k in bk if k not in keys_to_delete)
                tensors_to_keep = [t for k, t in zip(bk, tensor.unbind(self.batch_dim)) if k in keys_to_keep]
                for k, tensor in zip(keys_to_keep, tensors_to_keep):
                    super().__setitem__(k, tensor.unsqueeze(self.batch_dim))

                self.individual_keys = self.individual_keys.union(keys_to_keep)
                super().__delitem__(bk)

        return None


class DefaultStateStore(StateStore):
    def __init__(self, *args, batch_dim: int, default_state: torch.Tensor, **kwargs):
        super().__init__(*args, batch_dim=batch_dim, **kwargs)

    def __getitem__(self, keys: Union[Any, Tuple[Any]]):
        raise NotImplementedError()


StreamTensorOrPackedSequence = Union[StreamTensor, torch.nn.utils.rnn.PackedSequence]


def rnn_pre_hook(
    self, inputs: Union[StreamTensorOrPackedSequence, Tuple[StreamTensorOrPackedSequence, Optional[torch.Tensor]]]
):
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

    input_is_packed = isinstance(x, torch.nn.utils.rnn.PackedSequence)
    input_is_stream_tensor = isinstance(x.data, StreamTensor) if input_is_packed else isinstance(x, StreamTensor)
    if not self.streaming:
        state_is_stream_tensor = isinstance(state, StreamTensor) if state is not None else False
        if input_is_stream_tensor or state_is_stream_tensor:
            raise RuntimeError("Using StreamTensors in offline mode might result in unexpected behavior.")
        return inputs

    if not input_is_stream_tensor:
        raise RuntimeError("The input is expected to be StreamTensor when in online mode.")

    # Make the input a PackedSequence. TODO (JDH): Don't do this if input is a non-padded batch.
    if input_is_packed:
        ids = x.data.meta.ids
    else:
        # Pack the sequence and unsort the ids to either i) match the given `state` or ii) return the `state` from the 
        # `hidden_state_store` in the unsorted order. This will be sorted with `x.sorted_indices` inside the recurrent 
        # module's `forward` method.
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x.meta.lengths, batch_first=self.batch_first, enforce_sorted=False)
        ids = tuple(x.data.meta.ids[i] for i in x.unsorted_indices)  

    x.data.meta._temp_input_was_packed = input_is_packed

    if isinstance(state, StreamTensor):
        state = state.tensor()

    batch_size = len(ids)
    all_ids_in_state = ids in self.hidden_state_store
    if not all_ids_in_state:
        ids_without_state = tuple((i, _id) for i, _id in enumerate(ids) if _id not in self.hidden_state_store)
        if len(ids_without_state) == batch_size and state is not None:
            # No ids have state but a custom state is given.
            # Use the custom state for all examples.
            pass
        elif len(ids_without_state) == batch_size and state is None:
            # No ids have state and no custom state is given.
            # Use the module-native default state by passing None state.
            state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.data.device, dtype=x.data.dtype)
        elif len(ids_without_state) < batch_size and state is None:
            # Some ids have state but no custom state is given.
            # Write module-native default state into the StateStore for ids without state and gather contiguous state.
            default_state = torch.zeros(self.num_layers, len(ids_without_state), self.hidden_size, device=x.data.device, dtype=x.data.dtype)
            ids_without_state = tuple(_id for i, _id in ids_without_state)
            self.hidden_state_store[ids_without_state] = default_state
            state = self.hidden_state_store[ids]
        else:
            # Some ids have state but a state is given too.
            # Write custom state into StateStore for ids with out state and gather contiguous state.
            # TODO (JDH): Speed this up by using torch.select_scatter or similar directly on the given state.
            for i, _id in ids_without_state:
                self.hidden_state_store[_id] = state[:, i, :].unsqueeze(1)
            state = self.hidden_state_store[ids]
            # # Write default_state into the state tensor at the batch ids that do not have state.
            # index = torch.as_tensor(ids_without_state, device=x.device)
            # dim = 0 if self.batch_first else 1
            # torch.select_scatter(state, default_state, dim=dim, index=index)
    else:
        # All ids have state. Return the state.
        state = self.hidden_state_store[ids]

    return x, state


def rnn_post_hook(self, inputs, outputs):
    if not self.streaming:
        return outputs

    print("post hook")
    input, in_state = get_tensor_and_state(inputs)
    output, out_state = get_tensor_and_state(outputs)
    
    meta = input.meta if isinstance(input, StreamTensor) else input.data.meta
    if not meta._temp_input_was_packed:
        output.data.meta = input.data.meta
        output, lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        meta = output.meta

    # Ensure out_state is a proper StreamTensor.
    if isinstance(out_state, StreamTensor):
        out_state = out_state.tensor()

    # Store hidden states for this batch
    self.hidden_state_store[meta.ids] = out_state

    # Delete hidden states of ending files.
    ending_ids = tuple(_id for _id, eos in zip(meta.ids, meta.eos) if eos)
    if ending_ids:
        del self.hidden_state_store[ending_ids]

    return output, out_state


def patch_rnn(module):
    add_streaming_modes(module)

    # Add a dictionary for storing previous hidden states of shape (D * num_layers, N, H).
    module.hidden_state_store = StateStore(batch_dim=1)

    # Register pre_hook and post_hook.
    module.register_forward_pre_hook(rnn_pre_hook)
    module.register_forward_hook(rnn_post_hook)
    return module
