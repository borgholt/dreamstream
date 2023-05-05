import dataclasses
from typing import Callable, List, Optional
import torch


@dataclasses.dataclass()
class StreamState:
    """State associated with a batch of streamed input tensors."""

    ids: list[str]
    is_first: torch.BoolTensor
    is_last: torch.BoolTensor
    lengths: torch.IntTensor
    chunk_index: torch.IntTensor
    num_chunks: Optional[torch.IntTensor] = None


class StreamTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x: torch.Tensor, stream_state: StreamState, *args, **kwargs) -> torch.Tensor:
        # TODO (JDH): Should this return a StreamTensor instead of a torch.Tensor?
        return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, x: torch.Tensor, stream_state: StreamState, *args, **kwargs):
        # TODO (JDH): Should this call super().__init__ and take in the same args as torch.Tensor?
        self._stream_state = stream_state

    @classmethod
    def __torch_function__(cls, func: Callable, types: List[torch.Tensor], args=(), kwargs=None):
        stream_states = [x._stream_state for x in args if isinstance(x, StreamTensor)]
        # TODO: If more than one, assert that stream_states are identical, or raise error
        out = super().__torch_function__(func, types, args, kwargs)
        if isinstance(out, torch.Tensor):
            return StreamTensor(out, stream_state=stream_states[0])
        return out

    @property
    def stream_state(self) -> StreamState:
        return self._stream_state

    @stream_state.setter
    def stream_state(self, s: StreamState):
        self._stream_state = s
