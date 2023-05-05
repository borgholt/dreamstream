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
