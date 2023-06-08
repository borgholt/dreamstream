import pytest

import torch

# TODO (JDH): Tests will fail if we don't also import this, since nothing gets overridden.
import dreamstream.overrides  # noqa: F401

from dreamstream.tensor import StreamTensor, stream_tensor, stream_metadata, LENGTH, BATCH
from dreamstream.func_coverage import (
    DECOUPLE_FUNCTIONS,
    FLAT_OVERRIDABLE_FUNCTIONS,
    OVERRIDDEN_FUNCTIONS,
    RECOUPLE_FUNCTIONS,
    UNSUPPORTED_FUNCTIONS,
    VALID_FUNCTIONS,
)


@pytest.fixture()
def stream_tensor_3d():
    meta = stream_metadata(
        ids=["first", "middle", "last"], sos=[True, False, False], eos=[False, False, True], lengths=[3, 3, 2]
    )
    tensor = stream_tensor(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]],
        meta,
        names=(BATCH, "F", LENGTH),
    )  # (batch, feature, length) = (3, 2, 3)
    return tensor


## Instantiation


@pytest.mark.parametrize(
    "data",
    [
        [[1, 2, 3], [4, 5, 6]],  # list
        torch.tensor([[1, 2, 3], [4, 5, 6]]),  # tensor
        torch.tensor([[1, 2, 3], [4, 5, 6]], names=("dim1", "dim2")),  # named tensor
    ],
)
def test_instantiate_stream_tensor(data):
    """Test that we can instantiate a StreamTensor from different kinds of data."""
    meta = stream_metadata(ids=["a", "b"], sos=[True, False], eos=[False, True], lengths=[3, 3])
    tensor = stream_tensor(data, meta, names=(BATCH, LENGTH))

    assert isinstance(tensor, StreamTensor)
    assert tensor.meta == meta
    assert tensor.names == (BATCH, LENGTH)
    assert torch.equal(tensor.tensor().rename(None), torch.tensor(data).rename(None))


## Valid, coupled and recoupled functions, unsupported functions and function coverage

TEST_KWARGS_VALID_FUNCTIONS = {
    torch.Tensor.__repr__: dict(),
}

TEST_KWARGS_DECOUPLE_FUNCTIONS = {
    torch.argmax: dict(dim=-1),
    torch.argmin: dict(dim=-1),
    torch.argsort: dict(dim=-1),
}

TEST_KWARGS_RECOUPLE_FUNCTIONS = {
    torch.adjoint: dict(),
    torch.sigmoid: dict(),
    torch.Tensor.sigmoid: dict(),
    torch.transpose: dict(dim0=0, dim1=1),
}

TEST_SKIP_EQUALITY_CHECK = {torch.Tensor.__repr__}


def test_valid_coupled_recoupled_functions(stream_tensor_3d):
    FUNCTIONS = VALID_FUNCTIONS | DECOUPLE_FUNCTIONS | RECOUPLE_FUNCTIONS
    KWARGS = TEST_KWARGS_VALID_FUNCTIONS | TEST_KWARGS_DECOUPLE_FUNCTIONS | TEST_KWARGS_RECOUPLE_FUNCTIONS

    assert len(FUNCTIONS) == len(KWARGS), "The number of functions and test arguments must match."

    failed = []
    for f in FUNCTIONS:
        try:
            kwargs = KWARGS[f]
            stream_tensor_out = f(stream_tensor_3d, **kwargs)
            torch_tensor_out = f(stream_tensor_3d.tensor(), **kwargs)

            if f in TEST_SKIP_EQUALITY_CHECK:
                continue

            if isinstance(torch_tensor_out, torch.Tensor):
                stream_tensor_out = stream_tensor_out.rename(None)
                torch_tensor_out = torch_tensor_out.rename(None)
                assert torch.equal(stream_tensor_out, torch_tensor_out)
            else:
                assert stream_tensor_out == torch_tensor_out

        except Exception as e:
            failed.append((f, e))

    if any(failed):
        failed_str = "\n\n".join([f"{f.__name__}: {e}" for f, e in failed])
        raise AssertionError(f"The following functions claimed to be valid, were not:\n{failed_str}")


# def test_unsupported_functions(stream_tensor_3d):
#     failed = []
#     for f in UNSUPPORTED_FUNCTIONS:
#         try:
#             f(stream_tensor_3d)
#         except Exception as e:
#             failed.append((f, e))

#     if not all(failed):
#         failed_str = "\n".join([f"{f.__name__}: {e}" for f, e in failed])
#         raise AssertionError(f"The following functions claimed to be invalid, were not:\n{failed_str}")


def test_function_coverage():
    """Test that we have covered all functions in torch.nn.functional."""
    num_overridden = len(OVERRIDDEN_FUNCTIONS)
    num_valid = len(VALID_FUNCTIONS)
    num_invalid = len(UNSUPPORTED_FUNCTIONS)
    num_total = num_overridden + num_valid + num_invalid

    # fraction_working = (num_overridden + num_valid) / num_total

    msg = "Total number of functions must be the number of overridable functions plus any dunder methods."
    assert num_total >= len(FLAT_OVERRIDABLE_FUNCTIONS), msg
    # assert fraction_working > 0.8, f"Only {fraction_working*100:.1f} % of torch functions are covered (req >80%)."


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
    assert s1.names == ("F", LENGTH)


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
    s1 = stream_tensor_3d[torch.tensor([False, True, True])]

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
    - the lengths should be minus the number of non-padding tensor elements that are removed.
    """
    s2 = stream_tensor_3d[:, :, 1:]  # remove first length index

    assert isinstance(s2, StreamTensor)
    assert s2.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s2.meta.sos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s2.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s2.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all)

    s3 = stream_tensor_3d[:, :, :-1]  # remove last length index

    assert isinstance(s3, StreamTensor)
    assert s3.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s3.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s3.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s3.meta.lengths, torch.tensor([2, 2, 2]))  # minus 1 for all but "last" since it was padding

    s4 = stream_tensor_3d[:, :, 1:-1]  # remove first and last length index

    assert isinstance(s4, StreamTensor)
    assert s4.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s4.meta.sos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s4.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s4.meta.lengths, torch.tensor([1, 1, 1]))  # minus 2 for all but "last" since it was padding

    s5 = stream_tensor_3d[:, :, :-2]  # remove two last length indices

    assert isinstance(s5, StreamTensor)
    assert s5.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s5.meta.sos, torch.tensor([True, False, False]))  # changed to False
    assert torch.equal(s5.meta.eos, torch.tensor([False, False, False]))
    assert torch.equal(s5.meta.lengths, torch.tensor([1, 1, 1]))  # minus 2 for all but "last" since it was padding

    s6 = stream_tensor_3d[:, :, ::2]  # remove every other length index from start to end

    assert isinstance(s6, StreamTensor)
    assert s6.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s6.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s6.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s6.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all


def test_length_indexing_list_tuple(stream_tensor_3d):
    """"""
    s1 = stream_tensor_3d[:, :, [1, 2]]  # remove first length index

    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s1.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all
    assert s1.names == (BATCH, "F", LENGTH)

    s2 = stream_tensor_3d[:, :, [0, 2]]  # remove middle length index

    assert isinstance(s2, StreamTensor)
    assert s2.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s2.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s2.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s2.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all
    assert s2.names == (BATCH, "F", LENGTH)

    s3 = stream_tensor_3d[:, :, [0, 1]]  # remove last length index

    assert isinstance(s3, StreamTensor)
    assert s3.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s3.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s3.meta.eos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s3.meta.lengths, torch.tensor([2, 2, 2]))  # minus 1 for all but "last" since it was padding


def test_length_indexing_1d_inttensor(stream_tensor_3d):
    """"""
    s1 = stream_tensor_3d[:, :, torch.tensor([1, 2])]  # remove first length index

    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s1.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all

    s2 = stream_tensor_3d[:, :, torch.tensor([0, 2])]  # remove middle length index

    assert isinstance(s2, StreamTensor)
    assert s2.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s2.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s2.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s2.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all

    s3 = stream_tensor_3d[:, :, torch.tensor([0, 1])]  # remove last length index

    assert isinstance(s3, StreamTensor)
    assert s3.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s3.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s3.meta.eos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s3.meta.lengths, torch.tensor([2, 2, 2]))  # minus 1 for all but "last" since it was padding


def test_length_indexing_1d_booltensor(stream_tensor_3d):
    """Test that we can index a StreamTensor with a bool tensor (mask)."""
    # Indexing with a bool tensor on the length dim should correctly adjust the meta lengths and sos/eos.
    s1 = stream_tensor_3d[:, :, torch.tensor([False, True, True])]  # remove first length index

    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s1.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all
    assert s1.names == (BATCH, "F", LENGTH)

    s2 = stream_tensor_3d[:, :, torch.tensor([True, True, False])]  # remove last length index

    assert isinstance(s2, StreamTensor)
    assert s2.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s2.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s2.meta.eos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s2.meta.lengths, torch.tensor([2, 2, 2]))  # minus 1 for all but "last" since this was padding.
    assert s2.names == (BATCH, "F", LENGTH)


def test_indexing_ellipsis(stream_tensor_3d):
    """Test that we can index a StreamTensor with an ellipsis."""
    s1_1 = stream_tensor_3d[0, ...]  # keep only first batch example

    assert isinstance(s1_1, StreamTensor)
    assert s1_1.meta.ids == ["first"]
    assert torch.equal(s1_1.meta.sos, torch.tensor([True]))
    assert torch.equal(s1_1.meta.eos, torch.tensor([False]))
    assert torch.equal(s1_1.meta.lengths, torch.tensor([3]))
    assert s1_1.names == ("F", LENGTH)

    s1_2 = stream_tensor_3d[0, ..., :]  # keep only first batch example

    assert isinstance(s1_2, StreamTensor)
    assert s1_2.meta.ids == ["first"]
    assert torch.equal(s1_2.meta.sos, torch.tensor([True]))
    assert torch.equal(s1_2.meta.eos, torch.tensor([False]))
    assert torch.equal(s1_2.meta.lengths, torch.tensor([3]))
    assert s1_1.names == ("F", LENGTH)

    s1_3 = stream_tensor_3d[0, :, ...]  # keep only first batch example

    assert isinstance(s1_3, StreamTensor)
    assert s1_3.meta.ids == ["first"]
    assert torch.equal(s1_3.meta.sos, torch.tensor([True]))
    assert torch.equal(s1_3.meta.eos, torch.tensor([False]))
    assert torch.equal(s1_3.meta.lengths, torch.tensor([3]))
    assert s1_1.names == ("F", LENGTH)

    s1_4 = stream_tensor_3d[0, ..., ...]  # keep only first batch example

    assert isinstance(s1_4, StreamTensor)
    assert s1_4.meta.ids == ["first"]
    assert torch.equal(s1_4.meta.sos, torch.tensor([True]))
    assert torch.equal(s1_4.meta.eos, torch.tensor([False]))
    assert torch.equal(s1_4.meta.lengths, torch.tensor([3]))
    assert s1_1.names == ("F", LENGTH)

    s3 = stream_tensor_3d[..., 0]  # keep only first length index

    assert isinstance(s3, StreamTensor)
    assert s3.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s3.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s3.meta.eos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s3.meta.lengths, torch.tensor([1, 1, 1]))  # set to 1
    assert s3.names == (BATCH, "F")

    s4 = stream_tensor_3d[..., 0, ...]  # keep only first feature dim

    assert isinstance(s4, StreamTensor)
    assert s4.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s4.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s4.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s4.meta.lengths, torch.tensor([3, 3, 2]))


def test_batch_and_feature_indexing_2d_booltensor(stream_tensor_3d):
    """Test that we can index a StreamTensor with a 2d bool tensor along the batch and feature dimensions."""
    indices = torch.tensor([[True, True], [False, True], [True, False]])  # Keep only some features
    s1 = stream_tensor_3d[indices]

    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s1.meta.lengths, torch.tensor([3, 3, 2]))
    assert s1.shape == (4, 3)
    assert s1.names == (dreamstream.overrides.join_dim_names(BATCH, "F"), LENGTH)


def test_length_and_feature_indexing_2d_booltensor(stream_tensor_3d):
    """Test that we can index a StreamTensor with a 2d bool tensor along the length and feature dimensions."""
    indices = torch.tensor(
        [[True, True, True], [True, False, False]]
    )  # keep all lenghts of one feature and some of other
    s1 = stream_tensor_3d[:, indices]

    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s1.meta.lengths, torch.tensor([3, 3, 2]))
    assert s1.names == (BATCH, dreamstream.overrides.join_dim_names("F", LENGTH))

    indices = torch.tensor([[True, False, False], [True, False, False]])  # remove two last length steps of each feature
    s2 = stream_tensor_3d[:, indices]

    assert isinstance(s2, StreamTensor)
    assert s2.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s2.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s2.meta.eos, torch.tensor([False, False, False]))
    assert torch.equal(s2.meta.lengths, torch.tensor([1, 1, 1]))  # changed to 1
    assert s2.names == (BATCH, dreamstream.overrides.join_dim_names("F", LENGTH))


def test_batch_and_length_indexing_2d_booltensor(stream_tensor_3d):
    """Test that we can index a StreamTensor with a 2d bool tensor along the length and feature dimensions."""
    # sos=[True, False, False], eos=[False, False, True],
    indices = torch.tensor([[True, True, True], [True, False, False], [True, True, True]])  # (B, T) = (3, 3)
    s1 = stream_tensor_3d.permute(0, 2, 1)  # (B, F, T) -> (B, T, F)
    s1 = s1[indices]  # (B, T, F) -> (B_T, F)

    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s1.meta.lengths, torch.tensor([3, 1, 2]))  # changed to 1
    assert s1.names == (dreamstream.overrides.join_dim_names(BATCH, LENGTH), "F")

    # Remove middle example altogether
    indices = torch.tensor([[True, True, True], [False, False, False], [True, True, True]])  # (B, T) = (3, 3)
    s2 = stream_tensor_3d.permute(0, 2, 1)  # (B, F, T) -> (B, T, F)
    s2 = s2[indices]  # (B, T, F) -> (B_T, F)

    assert isinstance(s2, StreamTensor)
    assert s2.meta.ids == ["first", "last"]
    assert torch.equal(s2.meta.sos, torch.tensor([True, False]))
    assert torch.equal(s2.meta.eos, torch.tensor([False, True]))
    assert torch.equal(s2.meta.lengths, torch.tensor([3, 2]))
    assert s2.names == (dreamstream.overrides.join_dim_names(BATCH, LENGTH), "F")

    indices = torch.tensor([[False, True, True], [True, False, False], [True, False, False]])  # (B, T) = (3, 3)
    s3 = stream_tensor_3d.permute(0, 2, 1)  # (B, F, T) -> (B, T, F)
    s3 = s3[indices]  # (B, T, F) -> (B_T, F)

    assert isinstance(s3, StreamTensor)
    assert s3.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s3.meta.sos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s3.meta.eos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s3.meta.lengths, torch.tensor([2, 1, 1]))  # changed to 1
    assert s3.names == (dreamstream.overrides.join_dim_names(BATCH, LENGTH), "F")

    # Transposed version of s3
    indices = indices.transpose(0, 1)  # (B, T) -> (T, B)
    s4 = stream_tensor_3d.permute(2, 0, 1)  # (B, F, T) -> (T, B, F)
    s4 = s4[indices]  # (B, T, F) -> (B_T, F)

    assert isinstance(s4, StreamTensor)
    assert s4.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s4.meta.sos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s4.meta.eos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s4.meta.lengths, torch.tensor([2, 1, 1]))  # changed to 1
    assert s4.names == (dreamstream.overrides.join_dim_names(LENGTH, BATCH), "F")


def test_batch_and_length_indexing_3d_booltensor(stream_tensor_3d):
    """Test that we can index a StreamTensor with a 3d bool tensor along the batch, feature and length dimensions."""
    indices = torch.tensor(
        [
            [
                [True, True, True],  # first example, keep all steps in F1 but only first step in F2
                [True, False, False],
            ],
            [
                [True, False, False],  # second example, keep all steps in F1 but only first step in F2
                [True, False, False],
            ],
            [
                [True, False, False],  # third example, keep all steps in F1 but only first step in F2
                [True, False, False],
            ],
        ]
    )  # (B, T) = (3, 3)
    s1 = stream_tensor_3d[indices]  # (B, T, F) -> (B_T, F)

    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s1.meta.eos, torch.tensor([False, False, False]))  # changed to False
    assert torch.equal(s1.meta.lengths, torch.tensor([3, 1, 1]))  # changed to 1
    assert s1.names == (dreamstream.overrides.join_dim_names(BATCH, "F", LENGTH),)


def test_feature_indexing_2d_inttensor(stream_tensor_3d):
    indices = torch.tensor([[0, 1], [1, 1]])

    s1 = stream_tensor_3d[:, indices]

    assert isinstance(s1, StreamTensor)
    assert s1.meta.ids == ["first", "middle", "last"]
    assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
    assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
    assert torch.equal(s1.meta.lengths, torch.tensor([3, 3, 2]))
    assert s1.names == (BATCH, None, "F", LENGTH)
    assert s1.shape == (3, 2, 2, 3)


def test_batch_indexing_2d_inttensor(stream_tensor_3d):
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
    with pytest.raises(IndexError):
        stream_tensor_3d[indices]


def test_length_indexing_2d_inttensor(stream_tensor_3d):
    indices = torch.tensor([[0, 1], [1, 1]])
    with pytest.raises(IndexError):
        stream_tensor_3d[..., indices]


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
    assert torch.equal(s1.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all)
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
