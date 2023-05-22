import uuid
import dataclasses
from typing import Optional, Union, List, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from dreamstream.tensor import StreamTensor, StreamState
from dreamstream.utils.flags import BATCH, LENGTH


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
            
#             # Create stream state.
#             is_first = torch.full((tensor.size(1),), i == 0, dtype=torch.bool)
#             is_last = num_chunks == (i + 1)
#             chunk_lengths = torch.clip(lengths - (i * chunk_size), min=0, max=chunk_size)
#             stream_state = StreamState(ids, is_first, is_last, chunk_lengths)
            
#             # Discard empty sequences.
#             mask = chunk_lengths > 0
#             stream_state.filter(mask)
#             chunk = chunk[:, mask].refine_names(*names) # TODO: After implementing masked select, call refine names earlier.
            
#             # Convert to stream tensor chunk.
#             chunk = StreamTensor(chunk, stream_state)
#             self.append(chunk)
        
#         self.chunk_size = chunk_size
#         self.names = names
#         self.lengths = lengths
#         self.num_chunks = num_chunks

class Chunks(list):
    
    def __init__(
        self,
        chunks: List[StreamTensor],
        lengths: torch.Tensor,
        iterator_device: torch.device = None
    ):
        # TODO: Ensure that all tensors have same names and ids (or a subset of the previous chunk).
        self.ids = chunks[0].ids
        self.names = chunks[0].names
        self.lengths = lengths
    
    @property
    def num_chunks(self):
        return len(self)    
        
class OutputCollector(dict):
        
    def update(self, *stream_tensor: StreamTensor):
        
        for t in stream_tensor:
            length_dim = t.names.index(LENGTH)
            if BATCH in t.names:
                batch_dim = t.names.index(BATCH)
                for x in t.unpad_sequence():
                  self._update_unary(x, dim=length_dim)  
            else:
                assert stream_tensor.stream_state.size() == 1, "The tensor has no batch dimension, but state has multiple elements."
                self._update_unary(t, dim=length_dim)
                
    def _update_unary(self, stream_tensor: StreamTensor, dim: int):
        
        _id = stream_tensor.stream_state.id[0]
        if _id in self:
            self[_id] = torch.cat([self[_id], stream_tensor], dim=dim)
        else:
            self[_id] = stream_tensor
                