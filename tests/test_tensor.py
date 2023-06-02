import pytest
import inspect

import torch

import dreamstream.overrides  # TODO (JDH): Tests will fail if we don't also import this, since nothing gets overridden.

from dreamstream.tensor import OVERRIDDEN_FUNCTIONS, StreamTensor, StreamMetadata, stream_tensor, as_stream_tensor, stream_metadata, LENGTH, BATCH
from dreamstream.func_coverage import FLAT_OVERRIDABLE_FUNCTIONS, VALID_FUNCTIONS, UNSUPPORTED_FUNCTIONS


@pytest.fixture()
def stream_tensor_3d():
    meta = stream_metadata(ids=["first", "middle", "last"], sos=[True, False, False], eos=[False, False, True], lengths=[3, 3, 2])
    tensor = stream_tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]], meta, names=(BATCH, "F", LENGTH))
    return tensor


## Instantiation

@pytest.mark.parametrize("data", [
    [[1, 2, 3], [4, 5, 6]], # list
    torch.tensor([[1, 2, 3], [4, 5, 6]]), # tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]], names=("dim1", "dim2")), # named tensor
])
def test_instantiate_stream_tensor(data):
    """Test that we can instantiate a StreamTensor from different kinds of data."""
    meta = stream_metadata(ids=["a", "b"], sos=[True, False], eos=[False, True], lengths=[3, 3])
    tensor = stream_tensor(data, meta, names=(BATCH, LENGTH))

    assert isinstance(tensor, StreamTensor)
    assert tensor.meta == meta
    assert tensor.names == (BATCH, LENGTH)
    assert torch.equal(tensor.tensor().rename(None), torch.tensor(data).rename(None))


## Valid functions

def test_valid_functions(stream_tensor_3d):
    failed = []
    for f in VALID_FUNCTIONS:
        try:
            # kwargs = inspect.signature(f).parameters.keys()
            f(stream_tensor_3d)
        except Exception as e:
            failed.append((f, e))

    if any(failed):
        failed_str = "\n".join([f"{f.__name__}: {e}" for f, e in failed])
        raise AssertionError(f"The following functions claimed to be valid, were not:\n{failed_str}")


def test_invalid_functions(stream_tensor_3d):
    failed = []
    for f in UNSUPPORTED_FUNCTIONS:
        try:
            f(stream_tensor_3d)
        except Exception as e:
            failed.append((f, e))

    if not all(failed):
        failed_str = "\n".join([f"{f.__name__}: {e}" for f, e in failed])
        raise AssertionError(f"The following functions claimed to be invalid, were not:\n{failed_str}")
    

def test_function_coverage():
    """Test that we have covered all functions in torch.nn.functional."""
    num_overridden = len(OVERRIDDEN_FUNCTIONS)
    num_valid = len(VALID_FUNCTIONS)
    num_invalid = len(UNSUPPORTED_FUNCTIONS)
    num_total = num_overridden + num_valid + num_invalid

    fraction_working = (num_overridden + num_valid) / num_total

    assert num_total == len(FLAT_OVERRIDABLE_FUNCTIONS)
    assert fraction_working > 0.8, f"Only {fraction_working*100:.1f} % of functions are covered (req >80%)."


## Indexing

def test_feature_indexing_integer(stream_tensor_3d):
    """Indexing with an integer on the feature dim should remove the feature dim but not change the meta."""
    s5 = stream_tensor_3d[:, 0, :]
    
    assert isinstance(s5, StreamTensor)
    assert s5.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s5.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s5.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s5.meta.lengths, torch.tensor([3, 3, 2]))
    assert s5.names == (BATCH, LENGTH)


def test_batch_indexing_integer(stream_tensor_3d):
    """Indexing with an integer on the batch dim should remove the batch dim and change the meta to have only that one
    example's metadata."""
    s1 = stream_tensor_3d[0]

    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["first"]  # changed to only the first example
    assert torch.equal(s1.meta.sos, torch.tensor([True]))
    assert torch.equal(s1.meta.eos, torch.tensor([False]))
    assert torch.equal(s1.meta.lengths, torch.tensor([3]))
    assert s1.names == ("F", LENGTH,)


def test_batch_indexing_slice(stream_tensor_3d):
    """Indexing with a slice on the batch dim should remove relevant examples from the meta."""
    s1 = stream_tensor_3d[1:]
    
    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["middle", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([False, False]))
    assert torch.equal(s1.meta.eos, torch.tensor([False, True]))
    assert torch.equal(s1.meta.lengths, torch.tensor([3, 2]))
    assert s1.names == (BATCH, "F", LENGTH)


def test_batch_indexing_list_tuple(stream_tensor_3d):
    """Indexing with a list or tuple on the batch dim should remove relevant examples from the meta."""
    s1 = stream_tensor_3d[[0, 2]]
    
    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["first", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([True, False]))
    assert torch.equal(s1.meta.eos, torch.tensor([False, True]))
    assert torch.equal(s1.meta.lengths, torch.tensor([3, 2]))
    assert s1.names == (BATCH, "F", LENGTH)


def test_batch_indexing_booltensor(stream_tensor_3d):
    """Indexing with a bool tensor on the batch dim should remove relevant examples from the meta."""
    tensor = stream_tensor_3d
    
    # Indexing with a bool tensor on the batch dim should remove relevant examples from the meta.
    s1 = tensor[torch.tensor([False, True, True])]

    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["middle", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([False, False]))
    assert torch.equal(s1.meta.eos, torch.tensor([False, True]))
    assert torch.equal(s1.meta.lengths, torch.tensor([3, 2]))
    assert s1.names == (BATCH, "F", LENGTH)


def test_batch_indexing_inttensor(stream_tensor_3d):
    """Indexing with an int tensor on the batch dim should remove relevant examples from the meta."""
    tensor = stream_tensor_3d
    
    # Indexing with a bool tensor on the batch dim should remove relevant examples from the meta.
    s1 = tensor[torch.tensor([1, 2])]

    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["middle", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([False, False]))
    assert torch.equal(s1.meta.eos, torch.tensor([False, True]))
    assert torch.equal(s1.meta.lengths, torch.tensor([3, 2]))
    assert s1.names == (BATCH, "F", LENGTH)


def test_length_indexing_integer(stream_tensor_3d):
    """Indexing with an integer on the length dim should remove the length dim and change the meta to have lengths 1 or
    0 depending on padding, and sos only true when the index is 0, and eos only true when the index is the 
    last non-padding index or beyond (TODO (JDH): Maybe we want eos False if in padding?).
    """
    s2 = stream_tensor_3d[:, :, 0]  # first length index
    
    assert isinstance(s2, StreamTensor)
    assert s2.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s2.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s2.meta.eos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s2.meta.lengths, torch.tensor([1, 1, 1]))  # changed to 1
    assert s2.names == (BATCH, "F")

    s3 = stream_tensor_3d[:, :, 1]  # middle length index
    
    assert isinstance(s3, StreamTensor)
    assert s3.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s3.meta.sos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s3.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s3.meta.lengths, torch.tensor([1, 1, 1]))  # changed to 1
    assert s3.names == (BATCH, "F")

    s4 = stream_tensor_3d[:, :, -1]  # last length index

    assert isinstance(s4, StreamTensor)
    assert s4.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s4.meta.sos, torch.tensor([False, False, False]))  # change to False
    assert torch.equal(s4.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s4.meta.lengths, torch.tensor([1, 1, 0]))  # changed to 1 and 0
    assert s4.names == (BATCH, "F")


def test_length_indexing_slice(stream_tensor_3d):
    """Indexing with a slice on the length dim should correctly adjust the meta lengths and sos/eos.
    Specifically, 
    - if the slice starts after the first index, no examples should be first. 
    - if the slice ends before the last index, no examples should be last.
    - the lengths should be reduced by the number of non-padding tensor elements that are removed.
    """
    
    s2 = stream_tensor_3d[:, :, 1:]  # remove first length index

    assert isinstance(s2, StreamTensor)
    assert s2.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s2.meta.sos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s2.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s2.meta.lengths, torch.tensor([2, 2, 1]))  # reduced by 1 for all)
    
    s3 = stream_tensor_3d[:, :, :-1]  # remove last length index
    
    assert isinstance(s3, StreamTensor)
    assert s3.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s3.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s3.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s3.meta.lengths, torch.tensor([2, 2, 2]))  # reduced by 1 for all but "last" since this was padding.
    
    s4 = stream_tensor_3d[:, :, 1:-1]  # remove first and last length index

    assert isinstance(s4, StreamTensor)
    assert s4.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s4.meta.sos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s4.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s4.meta.lengths, torch.tensor([1, 1, 1]))  # reduced by 2 for all but "last" since this was padding.
    
    s5 = stream_tensor_3d[:, :, :-2]  # remove two last length indices

    assert isinstance(s5, StreamTensor)
    assert s5.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s5.meta.sos, torch.tensor([True, False, False]))  # changed to False
    assert torch.equal(s5.meta.eos, torch.tensor([False, False, False]))
    assert torch.equal(s5.meta.lengths, torch.tensor([1, 1, 1]))  # reduced by 2 for all but "last" since this was padding.

    s6 = stream_tensor_3d[:, :, ::2]  # remove every other length index from start to end
    
    assert isinstance(s6, StreamTensor)
    assert s6.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s6.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s6.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s6.meta.lengths, torch.tensor([2, 2, 1]))  # reduced by 1 for all but "last" since this was padding.


# def test_length_indexing_list_tuple(stream_tensor_3d):
#     """"""
#     s1 = stream_tensor_3d[:, :, [1, 2]]  # remove first length index
    
#     assert isinstance(s1, StreamTensor)
#     assert s1.meta.ids == ["first", "middle", "last"]
#     assert torch.equal(s1.meta.sos, torch.tensor([False, False, False]))  # changed to False
#     assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
#     assert torch.equal(s1.meta.lengths, torch.tensor([2, 2, 1]))  # reduced by 1 for all but "last" since this was padding.
#     assert s1.names == (BATCH, "F", LENGTH)
    
#     s2 = stream_tensor_3d[:, :, [0, 2]]  # remove middle length index
    
#     assert isinstance(s2, StreamTensor)
#     assert s2.meta.ids == ["first", "middle", "last"]
#     assert torch.equal(s2.meta.sos, torch.tensor([True, False, False]))
#     assert torch.equal(s2.meta.eos, torch.tensor([False, False, True]))
#     assert torch.equal(s2.meta.lengths, torch.tensor([2, 2, 1]))  # reduced by 1 for all but "last" since this was padding.
#     assert s2.names == (BATCH, "F", LENGTH)
    
#     s3 = stream_tensor_3d[:, :, [0, 1]]  # remove last length index
    
#     assert isinstance(s3, StreamTensor)
#     assert s3.meta.ids == ["first", "middle", "last"]
#     assert torch.equal(s3.meta.sos, torch.tensor([True, False, False]))
#     assert torch.equal(s3.meta.eos, torch.tensor([False, False, False]))  # changed to False
#     assert torch.equal(s3.meta.lengths, torch.tensor([2, 2, 2]))  # reduced by 1 for all but "last" since this was padding.


# def test_length_indexing_1d_inttensor(stream_tensor_3d):
#     """"""
#     raise NotImplementedError()


# def test_length_indexing_1d_booltensor(stream_tensor_3d):
#     """Test that we can index a StreamTensor with a bool tensor (mask)."""
#     tensor = stream_tensor_3d
    
#     # Indexing with a bool tensor on the batch dim should remove relevant examples from the meta.
#     s1 = tensor[torch.tensor([False, True, True])]

#     assert isinstance(s1, StreamTensor)
#     assert s1.meta.ids == ["middle", "last"]
#     assert torch.equal(s1.meta.sos, torch.tensor([False, False]))
#     assert torch.equal(s1.meta.eos, torch.tensor([False, True]))
#     assert torch.equal(s1.meta.lengths, torch.tensor([3, 2]))
#     assert s1.names == (BATCH, "F", LENGTH)
    
#     # Indexing with a bool tensor on the length dim should correctly adjust the meta lengths and sos/eos.
#     s2 = tensor[:, :, torch.tensor([False, True, True])]  # remove first length index    

#     assert isinstance(s2, StreamTensor)
#     assert s2.meta.ids == ["first", "middle", "last"]
#     assert torch.equal(s2.meta.sos, torch.tensor([False, False, False]))  # changed to False
#     assert torch.equal(s2.meta.eos, torch.tensor([False, False, True]))
#     assert torch.equal(s2.meta.lengths, torch.tensor([2, 2, 1]))  # reduced by 1 for all
#     assert s2.names == (BATCH, "F", LENGTH)
    
#     s3 = tensor[:, :, torch.tensor([True, True, False])]  # remove last length index

#     assert isinstance(s3, StreamTensor)
#     assert s3.meta.ids == ["first", "middle", "last"]
#     assert torch.equal(s3.meta.sos, torch.tensor([True, False, False]))
#     assert torch.equal(s3.meta.eos, torch.tensor([False, False, False]))  # changed to False
#     assert torch.equal(s3.meta.lengths, torch.tensor([2, 2, 2]))  # reduced by 1 for all but "last" since this was padding.
#     assert s3.names == (BATCH, "F", LENGTH)


def test_length_indexing_nd_inttensor(stream_tensor_3d):
    """"""
    indices = torch.tensor([[0, 1], [0, 1]])
    with pytest.raises(NotImplementedError):
        stream_tensor_3d[:, :, indices]


def test_length_indexing_nd_booltensor(stream_tensor_3d):
    """"""
    indices = torch.tensor([[False, True, True], [False, True, False]])
    with pytest.raises(NotImplementedError):
        stream_tensor_3d[:, indices]


def test_length_indexing_integer_multidimensional(stream_tensor_3d):
    """Indexing with an integer on the length dim should remove the length dim and change the meta to have lengths 1 or
    0 depending on padding, and sos only true when the index is 0, and eos only true when the index is the 
    last non-padding index or beyond (TODO (JDH): Maybe we want eos False if in padding?).
    This case also tests that we can simultaneously index the feature dimension without affecting the length indexing.
    """
    s1 = stream_tensor_3d[:, 0, 0]  # first length index and first feature index
    
    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s1.meta.eos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s1.meta.lengths, torch.tensor([1, 1, 1]))  # changed to 1
    assert s1.names == (BATCH,)


def test_length_indexing_slice_multidimensional(stream_tensor_3d):
    """Indexing with a slice on the length dim should correctly adjust the meta lengths and sos/eos. 
    This case also tests that we can simultaneously index the feature dimension without affecting the length indexing.
    """
    s1 = stream_tensor_3d[:, 0, 1:]  # remove first length index and first feature index
    
    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s1.meta.lengths, torch.tensor([2, 2, 1]))  # reduced by 1 for all)
    assert s1.names == (BATCH, LENGTH)


def test_batch_and_length_indexing_slice(stream_tensor_3d):
    """Indexing with a slice on the batch dim and a slice on the length dim should have the combined effect of both."""
    s1 = stream_tensor_3d[:-1, :, 1:]  # remove last batch index and first length index
    
    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["first", "middle"]
    assert torch.equal(s1.meta.sos, torch.tensor([False, False]))
    assert torch.equal(s1.meta.eos, torch.tensor([False, False]))
    assert torch.equal(s1.meta.lengths, torch.tensor([2, 2]))
    assert s1.names == (BATCH, "F", LENGTH)
