
from typing import Optional, Sequence, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from dreamstream.tensor import StreamTensor, StreamState
from dreamstream.utils.flags import BATCH, LENGTH

def pad_chunks(
    sequences,
    names: List[str],
    ids: List[str],
    is_first: Union[List[bool], torch.BoolTensor],
    is_last: Union[List[bool], torch.BoolTensor],
    batch_first: bool = False
) -> StreamTensor:
    
    # Why not allow a length argument? Because then lengths can be passed, which should be inferred.
    # Why not allow a state argument? Same as above.
    # Why not allow names to be inferred from the sequences? Because then permute won't work.
    
    if BATCH in names:
        raise ValueError("Sequences must not have a batch dimension.")
    if LENGTH not in names:
        raise ValueError("Sequences must have a length dimension.")
    
    length_index = names.index(LENGTH)
    if length_index != 0:
        new_ordering = (length_index,) + tuple(i for i in range(len(names)) if i != length_index)
        sequences = [t.permute(*new_ordering) for t in sequences]
        
    prefix = (BATCH, LENGTH) if batch_first else (LENGTH, BATCH)
    names = prefix + tuple(n for n in names if n != LENGTH)
    lengths = torch.tensor([t.size(0) for t in sequences])
    tensor = pad_sequence(sequences, batch_first=batch_first)
    tensor = tensor.rename(*names)
    
    state = StreamState(
        ids=ids,
        is_first=is_first,
        is_last=is_last,
        lengths=lengths
    )
    
    return StreamTensor(data=tensor, stream_state=state) 
                        
def pad_full_sequence(
    sequences,
    names: List[str],
    ids: List[str],
    batch_first: bool = False
) -> StreamTensor:

    is_first = torch.full((len(sequences),), True, dtype=torch.bool)
    is_last = torch.full((len(sequences),), True, dtype=torch.bool)
    
    return pad_chunks(
        sequences=sequences,
        names=names,
        ids=ids,
        is_first=is_first,
        is_last=is_last,
        batch_first=batch_first
    )