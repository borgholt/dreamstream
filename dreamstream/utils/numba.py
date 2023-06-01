from typing import Tuple
import numba
import numpy as np


@numba.jit(nopython=True)
def minmax(array) -> Tuple[int, int]:
    """Return the minimum and maximum values of an array."""
    maximum = array[0]
    minimum = array[0]
    for i in array[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i
    return (minimum, maximum)


@numba.jit(nopython=True)
def is_sorted_ascending(array) -> bool:
    """Return True if the array is sorted in ascending order."""
    for i in range(1, len(array)):
        if array[i] < array[i - 1]:
            return False
    return True


@numba.jit(nopython=True)
def is_sorted_descending(array) -> bool:
    """Return True if the array is sorted in descending order."""
    for i in range(1, len(array)):
        if array[i] > array[i - 1]:
            return False
    return True


@numba.jit(nopython=True)
def make_indices_positive_(array, max_length: int) -> None:
    """Make all indices in the array positive."""
    negative_indices = array < 0

    if negative_indices.any():
        array[negative_indices] = max_length + array[negative_indices]

    return array


@numba.jit(nopython=True)
def broadcasted_leq_sum(array1, array2) -> np.ndarray:
    """Return a 1D array of the number of elements in array1 that are less than or equal to each element in array2.

    This is more than 10x faster than the equivalent torch version on CPU: 
    > (self.lengths.unsqueeze(0) <= indices_t.unsqueeze(1)).sum(dim=0)
    """
    return np.sum(np.expand_dims(array1, axis=0) <= np.expand_dims(array2, axis=1), axis=0)


@numba.jit(nopython=True)
def update_lengths_from_list_of_indices(lengths, indices) -> None:
    """For each element in lengths, return the number of elements in indices that are less than that element.

    This is used to return an updated lengths tensor after a list of indices has been used to select elements from a
    tensor.

    This is more than 10x faster than a non-compiled torch method on CPU.
    """
    return np.sum(np.expand_dims(lengths, axis=0) > np.expand_dims(indices, axis=1), axis=0)


@numba.jit(nopython=True)
def update_eos_from_integer(eos: np.ndarray, lengths: np.ndarray, index: int) -> None:
    """For each element in eos, return the number of elements in indices that are less than that element.

    This is used to return an updated eos tensor after a list of indices has been used to select elements from a
    tensor.

    This is 2x faster than the non-compiled torch method on CPU: `(eos & (lengths <= index + 1))`.
    """
    return eos & (lengths <= index + 1)  # TODO (JDH): Would `self.lengths < indices` be better?
