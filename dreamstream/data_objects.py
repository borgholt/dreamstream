import dataclasses
import uuid

from typing import Optional

import torch
import torchaudio


@dataclasses.dataclass()
class AudioSample:
    """Data for a sample of audio from a single audio file."""
    data: torch.Tensor
    is_first: bool
    is_last: bool
    length: int
    chunk_index: Optional[int] = None
    num_chunks: Optional[int] = None
    id: Optional[str] = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    file: Optional[str] = None
    file_metadata: Optional[torchaudio.backend.common.AudioMetaData] = None


@dataclasses.dataclass()
class StreamState:
    """State associated with a batch of streamed input tensors."""
    ids: list[str]
    is_first: torch.BoolTensor
    is_last: torch.BoolTensor
    lengths: torch.IntTensor
    chunk_index: torch.IntTensor
    num_chunks: Optional[torch.IntTensor] = None
