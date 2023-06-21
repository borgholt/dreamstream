import uuid
import dataclasses

from typing import Any, Optional, List, Tuple, Union
from pathlib import PosixPath

import torch
import torchaudio

from dreamstream.tensor import StreamTensor, StreamMetadata
from dreamstream.utils.flags import BATCH, LENGTH


@dataclasses.dataclass()
class StreamSample:
    """Data for a sample of audio from a single audio file."""

    data: torch.Tensor = dataclasses.field(repr=False)
    sos: bool = dataclasses.field(repr=True)
    eos: bool = dataclasses.field(repr=True)
    length: int = dataclasses.field(repr=True)
    chunk_index: Optional[int] = dataclasses.field(default=None, repr=True)
    num_chunks: Optional[int] = dataclasses.field(default=None, repr=True)
    id: Optional[str] = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    file: Optional[Union[str, PosixPath]] = dataclasses.field(default=None, repr=True)
    file_metadata: Optional[Any] = dataclasses.field(default=None, repr=True)


# class ChunkedList(list):

#     def __init__(
#         self,
#         data: Union[List[torch.Tensor], List[StreamTensor]],
#         ids: List[str],
#         chunk_size: int,
#         device: torch.device = None,
#         iterator_device: torch.device = None,
#         names: Union[List[str], Tuple[str]] = None
#     ):

#         # ensure that names are set correctly for all tensors
#         if names is not None:
#             data = [t.rename(*names) for t in data] # will overwrite names if already named
#         else:
#             names = data[0].names
#             if not all(t.names == names for t in data):
#                 raise ValueError("All tensors must have the same names.")
#         if BATCH in names:
#             raise ValueError("The data must not have a batch dimension.")
#         if LENGTH not in names:
#             raise ValueError("The data must have a length dimension.")

#         # move length dimension to front, if necessary
#         if names[0] != LENGTH:
#             align_names = (LENGTH,) + tuple(n for n in names if n != LENGTH)
#             data = [t.align_to(*align_names) for t in data]

#         # pad tensors to a batch of longform tensors with shape (L, B, ...)
#         names = (LENGTH, BATCH) + tuple(n for n in names if n != LENGTH)
#         lengths = torch.as_tensor([t.size(LENGTH) for t in data])
#         tensor = pad_sequence(data)
#         if device is not None:
#             tensor = tensor.to(device)

#         # Chunk up the longform tensor and store as list of stream tensors.
#         num_chunks = torch.ceil(lengths / chunk_size).to(torch.int)
#         chunks = torch.split(tensor, chunk_size, dim=0)
#         for i, chunk in enumerate(chunks):

#             # Create stream metadata.
#             sos = torch.full((tensor.size(1),), i == 0, dtype=torch.bool)
#             eos = num_chunks == (i + 1)
#             chunk_lengths = torch.clip(lengths - (i * chunk_size), min=0, max=chunk_size)
#             meta = StreamMetadata(ids, sos, eos, chunk_lengths)

#             # Discard empty sequences.
#             mask = chunk_lengths > 0
#             meta.filter(mask)

#               # TODO: After implementing masked select, call refine names earlier.
#             chunk = chunk[:, mask].refine_names(*names)

#             # Convert to stream tensor chunk.
#             chunk = StreamTensor(chunk, meta)
#             self.append(chunk)

#         self.chunk_size = chunk_size
#         self.names = names
#         self.lengths = lengths
#         self.num_chunks = num_chunks


class ChunkedList(list):
    def __init__(self, chunks: List[StreamTensor], meta: StreamMetadata):
        self += chunks
        self.meta = meta

    @property
    def num_chunks(self):
        return len(self)


class OutputCollector(dict):
    def __init__(self, *stream_tensor: StreamTensor):
        super().__init__()
        self.closed_entries = set()
        self.update(*stream_tensor)

    def update(self, *stream_tensors: Tuple[StreamTensor]):
        for t in stream_tensors:
            if BATCH in t.names:
                for x in t.unpad_sequence():
                    if x.meta.max_length > 0:
                        self._update_unary(x)
            else:
                assert t.meta.size() == 1, "The tensor has no batch dimension, but metadata has multiple elements."
                self._update_unary(t)

    def _update_unary(self, stream_tensor: StreamTensor):
        _id = stream_tensor.meta.ids[0]

        if _id in self.closed_entries:
            raise ValueError(f"The entry for {_id} has already been closed.")
        if stream_tensor.meta.eos.item():
            self.closed_entries.add(_id)

        if _id in self:
            assert not stream_tensor.meta.sos.item(), "The tensor is the first chunk."
            length_dim = stream_tensor.names.index(LENGTH)
            self[_id] = torch.cat([self[_id], stream_tensor], dim=length_dim)
        else:
            assert stream_tensor.meta.sos.item(), "The tensor is not the first chunk."
            self[_id] = stream_tensor
