import math

from typing import Union

import torch


def sequence_mask(
    seq_lens: Union[list, torch.Tensor],
    max_len: int = None,
    invert: bool = False,
    dtype: torch.dtype = torch.bool,
    device: torch.device = None,
):
    """
    Creates a binary sequence mask where all entries up to seq_lens are 1 and the remaining are 0.

    Args:
        seq_lens (Tensor): The sequence lengths from which to construct the mask. Should be shape N with dtype == int64.
        max_len (int): The temporal dimension of the sequence mask. If None, will use max of seq_lens.
        dtype (torch.dtype): The type of the mask. Default is torch.bool.
        invert (bool): If False, `m[i]` is `True` for `i < x_sl` and False for `i >= x_sl`.
                       If True, returns the inverse i.e. `~m`. Default is False.
    Returns:
        Tensor: The sequence mask of shape (N, T).
    """
    if isinstance(seq_lens, torch.Tensor):
        device = seq_lens.device if device is None else device
        if device != seq_lens.device:
            seq_lens = seq_lens.to(device)
    else:
        seq_lens = torch.tensor(seq_lens, device=device, dtype=int)

    T = max_len or math.ceil(seq_lens.max())

    step_ids = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

    if invert:
        seq_mask = step_ids >= seq_lens.unsqueeze(1)  # broadcast over batch, (N, T)
        return seq_mask.to(dtype)

    seq_mask = step_ids < seq_lens.unsqueeze(1)  # broadcast over batch, (N, T)
    return seq_mask.to(dtype)

