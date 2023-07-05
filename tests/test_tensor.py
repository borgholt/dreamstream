# import collections

import pytest
import torch

from dreamstream.tensor import StreamTensor, as_stream_tensor, stream_tensor, stream_metadata, LENGTH, BATCH
# from dreamstream.func_coverage import (
#     DECOUPLE_FUNCTIONS,
#     FLAT_OVERRIDABLE_FUNCTIONS,
#     CUSTOMIZED_FUNCTIONS,
#     RECOUPLE_FUNCTIONS,
#     UNSUPPORTED_FUNCTIONS,
#     VALID_FUNCTIONS,
# )
from dreamstream.overrides import join_dim_names


def test_data_2d():
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_data_3d():
    return [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]]


def test_meta_kwargs():
    return dict(
        ids=["first", "middle", "last"],
        sos=[True, False, False],
        eos=[False, False, True],
        lengths=[3, 3, 2],
    )


def stream_tensor_bl():
    """StreamTensor with shape (B, L) = (3, 3)"""
    test_meta = stream_metadata(**test_meta_kwargs())
    return stream_tensor(torch.as_tensor(test_data_2d()), test_meta, names=(BATCH, LENGTH))  # (3, 3)


def stream_tensor_bfl():
    """StreamTensor with shape (B, F, L) = (3, 2, 3)"""
    test_meta = stream_metadata(**test_meta_kwargs())
    return stream_tensor(test_data_3d(), test_meta, names=(BATCH, "F", LENGTH))  # (3, 2, 3)


def stream_tensor_lbf():
    """StreamTensor with shape (L, B, F) = (3, 3, 2)"""
    test_meta = stream_metadata(**test_meta_kwargs())
    data = torch.as_tensor(test_data_3d()).permute(2, 0, 1)  # (L, B, F) = (3, 3, 2)
    return stream_tensor(data, test_meta, names=(LENGTH, BATCH, "F"))


def stream_tensor_blf():
    """StreamTensor with shape (B, L, F) = (3, 3, 2)"""
    test_meta = stream_metadata(**test_meta_kwargs())
    data = torch.as_tensor(test_data_3d()).permute(0, 2, 1)  # (B, L, F) = (3, 3, 2)
    return stream_tensor(data, test_meta, names=(LENGTH, BATCH, "F"))


def stream_tensor_bfld():
    """ "StreamTensor with shape (B, F, L, D) = (2, 3, 2, 3)"""
    test_meta = stream_metadata(**test_meta_kwargs())
    test_tensor_data = torch.stack([torch.as_tensor(test_data_3d()), torch.as_tensor(test_data_3d())], dim=2)
    return stream_tensor(test_tensor_data, test_meta, names=(BATCH, "F", "D", LENGTH))  # (2, 3, 2, 3)


@pytest.fixture()
def stream_meta_kwargs_fixture():
    return test_meta_kwargs()


@pytest.fixture()
def stream_tensor_bfl_fixture():
    """StreamTensor with shape (B, F, L) = (3, 2, 3)"""
    return stream_tensor_bfl()


@pytest.fixture()
def stream_tensor_lbf_fixture():
    """StreamTensor with shape (L, B, F) = (3, 3, 2)"""
    return stream_tensor_lbf()


@pytest.fixture()
def stream_tensor_blf_fixture():
    """StreamTensor with shape (L, B, F) = (3, 3, 2)"""
    return stream_tensor_blf()


@pytest.fixture()
def stream_tensor_bfld_fixture():
    """StreamTensor with shape (B, F, L, D) = (2, 3, 2, 3)"""
    return stream_tensor_bfld()


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


class Inputs:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter((self.args, self.kwargs))


def to_torch_tensor_recursive(x):
    """Helper function similar to `decouple_recursive`."""
    if isinstance(x, StreamTensor):
        return x.tensor()
    elif isinstance(x, (list, tuple)):
        return type(x)(to_torch_tensor_recursive(xi) for xi in x)
    elif isinstance(x, dict):
        return type(x)((k, to_torch_tensor_recursive(v)) for k, v in x.items())
    return x


TEST_INPUTS_VALID_FUNCTIONS = {
    torch.Tensor.__repr__: Inputs(stream_tensor_bfl()),
    torch.Tensor.__str__: Inputs(stream_tensor_bfl()),
}

TEST_INPUTS_DECOUPLE_FUNCTIONS = {
    torch.argmax: Inputs(stream_tensor_bfl(), dim=-1),
    torch.argmin: Inputs(stream_tensor_bfl(), dim=-1),
    torch.argsort: Inputs(stream_tensor_bfl(), dim=-1),
}

TEST_INPUTS_RECOUPLE_FUNCTIONS = {
    torch.adjoint: Inputs(stream_tensor_bfl()),
    torch.sigmoid: Inputs(stream_tensor_bfl()),
    torch.Tensor.sigmoid: Inputs(stream_tensor_bfl()),
    torch.transpose: Inputs(stream_tensor_bfl(), dim0=0, dim1=1),
    torch.where: Inputs(stream_tensor_bfl() > 5, input=stream_tensor_bfl() - 1, other=stream_tensor_bfl()),
    torch.where: Inputs(stream_tensor_bfl() > 5, input=stream_tensor_bfl() - 1, other=2),
    torch.Tensor.where: Inputs(stream_tensor_bfl(), condition=stream_tensor_bfl() > 5, other=stream_tensor_bfl()),
    torch.scatter: Inputs(
        stream_tensor_bl(),
        dim=0,
        index=torch.tensor([[0, 1, 2], [0, 1, 2]]),
        src=stream_tensor_bl() * 2,
    ),
    torch.diagonal_scatter: Inputs(
        stream_tensor_bl(),
        src=stream_tensor_bl().tensor()[:, 0] * 2,
        offset=0,
        dim1=0,
        dim2=1,
    ),
    torch.select_scatter: Inputs(
        stream_tensor_bl(),
        src=stream_tensor_bl().tensor()[:, 0] * 2,
        dim=0,
        index=1,
    ),
    torch.slice_scatter: Inputs(
        stream_tensor_bl(),
        src=stream_tensor_bl().tensor() * 2,
        dim=0,
        start=0,
        end=3,
        step=1,
    ),
}


TEST_INPUTS_RECOUPLE_FUNCTIONS[torch.Tensor.scatter_] = TEST_INPUTS_RECOUPLE_FUNCTIONS[torch.scatter]
TEST_INPUTS_RECOUPLE_FUNCTIONS[torch.Tensor.scatter] = TEST_INPUTS_RECOUPLE_FUNCTIONS[torch.scatter]
TEST_INPUTS_RECOUPLE_FUNCTIONS[torch.Tensor.diagonal_scatter] = TEST_INPUTS_RECOUPLE_FUNCTIONS[torch.diagonal_scatter]
TEST_INPUTS_RECOUPLE_FUNCTIONS[torch.Tensor.select_scatter] = TEST_INPUTS_RECOUPLE_FUNCTIONS[torch.select_scatter]
TEST_INPUTS_RECOUPLE_FUNCTIONS[torch.Tensor.slice_scatter] = TEST_INPUTS_RECOUPLE_FUNCTIONS[torch.slice_scatter]


TEST_SKIP_EQUALITY_CHECK = {torch.Tensor.__repr__, torch.Tensor.__str__}


# def test_valid_coupled_recoupled_functions():
#     """Iterate over all valid, coupled and recoupled functions and check that they work as expected."""
#     INPUTS = collections.defaultdict(Inputs)
#     INPUTS.update(TEST_INPUTS_VALID_FUNCTIONS | TEST_INPUTS_DECOUPLE_FUNCTIONS | TEST_INPUTS_RECOUPLE_FUNCTIONS)

#     FUNCTIONS = VALID_FUNCTIONS | DECOUPLE_FUNCTIONS | RECOUPLE_FUNCTIONS

#     failed = []
#     for function in FUNCTIONS:
#         try:
#             args, kwargs = INPUTS[function]
#             stream_tensor_out = function(*args, **kwargs)
#             torch_tensor_out = function(*to_torch_tensor_recursive(args), **to_torch_tensor_recursive(kwargs))

#             if function in TEST_SKIP_EQUALITY_CHECK:
#                 continue

#             if isinstance(torch_tensor_out, torch.Tensor):
#                 stream_tensor_out = stream_tensor_out.rename(None)
#                 torch_tensor_out = torch_tensor_out.rename(None)
#                 assert torch.equal(stream_tensor_out, torch_tensor_out)
#             else:
#                 assert stream_tensor_out == torch_tensor_out

#         except Exception as e:
#             failed.append((function, e))

#     if any(failed):
#         failed_str = " - " + "\n - ".join([f"{f.__name__}: {e}" for f, e in failed])
#         err = f"The following {len(failed)} functions claimed to be valid, were not:\n{failed_str}"
#         raise AssertionError("\n" + err)


# def test_unsupported_functions(stream_tensor_bfl_fixture):
#     failed = []
#     for f in UNSUPPORTED_FUNCTIONS:
#         try:
#             f(stream_tensor_bfl_fixture)
#         except Exception as e:
#             failed.append((f, e))

#     if not all(failed):
#         failed_str = "\n".join([f"{f.__name__}: {e}" for f, e in failed])
#         raise AssertionError(f"The following functions claimed to be invalid, were not:\n{failed_str}")


# def test_function_coverage():
#     """Test that we have covered all functions in torch.nn.functional."""
#     num_customized = len(CUSTOMIZED_FUNCTIONS)
#     num_valid = len(VALID_FUNCTIONS)
#     num_invalid = len(UNSUPPORTED_FUNCTIONS)
#     num_total = num_customized + num_valid + num_invalid

#     fraction_working = (num_customized + num_valid) / num_total

#     msg = "Total number of functions must be the number of overridable functions plus any dunder methods."
#     assert num_total >= len(FLAT_OVERRIDABLE_FUNCTIONS), msg
#     assert fraction_working > 0.8, f"Only {fraction_working*100:.1f} % of torch functions are covered (req >80%)."


## Indexing functions


def assert_stream_tensor_and_meta_correct(stream_tensor, func, *args, ids, lengths, sos, eos, names, **kwargs):
    """Assert that a function returns a StreamTensor with the correct metadata when applied to a StreamTensor.

    The output is expected to be equal in value to the output when the function is applied to an identical torch.Tensor.
    """
    s = func(stream_tensor, *args, **kwargs)
    torch_tensor, args, kwargs = to_torch_tensor_recursive((stream_tensor, args, kwargs))
    t = func(torch_tensor, *args, **kwargs)
    assert isinstance(s, StreamTensor)
    assert torch.equal(s.tensor(), t)
    assert s.meta.ids == ids
    assert torch.equal(s.meta.sos, torch.tensor(sos))
    assert torch.equal(s.meta.eos, torch.tensor(eos))
    assert torch.equal(s.meta.lengths, torch.tensor(lengths))
    assert s.names == names


def assert_fallback_to_torch_tensor(stream_tensor, func, *args, **kwargs):
    """Assert that a function falls back to return a torch.Tensor when applied to a StreamTensor.

    The torch.Tensor is expected to have no names and be equal in value to the StreamTensor's tensor.
    """
    s = func(stream_tensor, *args, **kwargs)
    t = func(stream_tensor.tensor(), *args, **kwargs)
    assert isinstance(s, torch.Tensor) and not isinstance(s, StreamTensor)
    assert torch.equal(s, t)
    assert s.names == (None,) * s.ndim


class TestNarrow:
    def test_narrow_batch(self, stream_tensor_bfl_fixture):
        """Test `torch.narrow` on a StreamTensor when applied to batch, length, and feature dimensions."""
        assert_kwargs = dict(
            ids=["middle", "last"],
            lengths=[3, 2],
            sos=[False, False],
            eos=[False, True],
            names=(BATCH, "F", LENGTH),
        )
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.narrow,
            dim=0,
            start=1,
            length=2,
            **assert_kwargs,
        )

    def test_narrow_feature(self, stream_tensor_bfl_fixture):
        s = stream_tensor_bfl_fixture.narrow(dim=1, start=1, length=1)
        assert isinstance(s, StreamTensor)
        assert torch.equal(s.tensor(), stream_tensor_bfl_fixture.tensor().narrow(dim=1, start=1, length=1))
        assert s.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s.meta.lengths, torch.tensor([3, 3, 2]))
        assert s.names == (BATCH, "F", LENGTH)

    def test_narrow_length(self, stream_tensor_bfl_fixture):
        s = stream_tensor_bfl_fixture.narrow(dim=2, start=1, length=2)
        assert isinstance(s, StreamTensor)
        assert torch.equal(s.tensor(), stream_tensor_bfl_fixture.tensor().narrow(dim=2, start=1, length=2))
        assert s.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s.meta.sos, torch.tensor([False, False, False]))
        assert torch.equal(s.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s.meta.lengths, torch.tensor([2, 2, 1]))
        assert s.names == (BATCH, "F", LENGTH)


class TestGather:
    full_index = torch.tensor(
        [
            [
                [1, 0, 1],
                [0, 1, 1],
            ],
            [
                [0, 1, 1],
                [1, 0, 1],
            ],
            [
                [1, 0, 1],
                [0, 1, 1],
            ],
        ]
    )

    truncated_index = torch.tensor(
        [
            [
                [1, 0],
                [0, 1],
            ],
            [
                [0, 1],
                [1, 0],
            ],
        ]
    )

    def test_gather_batch(self, stream_tensor_bfl_fixture):
        """Test `torch.gather` on a StreamTensor when applied to batch, length, and feature dimensions."""
        with pytest.raises(IndexError):
            stream_tensor_bfl_fixture.gather(dim=0, index=self.full_index)

    def test_gather_feature(self, stream_tensor_bfl_fixture):
        s = stream_tensor_bfl_fixture.gather(dim=1, index=self.full_index)
        assert isinstance(s, StreamTensor)
        assert torch.equal(s.tensor(), stream_tensor_bfl_fixture.tensor().gather(dim=1, index=self.full_index))
        assert s.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s.meta.lengths, torch.tensor([3, 3, 2]))
        assert s.names == (BATCH, "F", LENGTH)

    def test_gather_feature_truncated(self, stream_tensor_bfl_fixture):
        s = stream_tensor_bfl_fixture.gather(dim=1, index=self.truncated_index)
        assert isinstance(s, StreamTensor)
        assert torch.equal(s.tensor(), stream_tensor_bfl_fixture.tensor().gather(dim=1, index=self.truncated_index))
        assert s.meta.ids == ["first", "middle"]
        assert torch.equal(s.meta.sos, torch.tensor([True, False]))
        assert torch.equal(s.meta.eos, torch.tensor([False, False]))
        assert torch.equal(s.meta.lengths, torch.tensor([2, 2]))
        assert s.names == (BATCH, "F", LENGTH)

    def test_gather_length(self, stream_tensor_bfl_fixture):
        with pytest.raises(IndexError):
            stream_tensor_bfl_fixture.gather(dim=2, index=self.full_index)


class TestTakeAlongDim:
    def test_take_along_batch(self, stream_tensor_bfl_fixture):
        """Test `torch.take_along_dim` on a StreamTensor when applied to batch, length, and feature dimensions."""
        with pytest.raises(IndexError):
            stream_tensor_bfl_fixture.take_along_dim(dim=0, indices=TestGather.full_index)

    def test_take_along_feature(self, stream_tensor_bfl_fixture):
        s = stream_tensor_bfl_fixture.take_along_dim(dim=1, indices=TestGather.full_index)
        assert isinstance(s, StreamTensor)
        assert torch.equal(
            s.tensor(), stream_tensor_bfl_fixture.tensor().take_along_dim(dim=1, indices=TestGather.full_index)
        )
        assert s.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s.meta.lengths, torch.tensor([3, 3, 2]))
        assert s.names == (BATCH, "F", LENGTH)

    def test_take_along_feature_truncated(self, stream_tensor_bfl_fixture):
        with pytest.raises(RuntimeError):
            stream_tensor_bfl_fixture.take_along_dim(dim=1, indices=TestGather.truncated_index)

    def test_take_along_length(self, stream_tensor_bfl_fixture):
        with pytest.raises(IndexError):
            stream_tensor_bfl_fixture.take_along_dim(dim=2, indices=TestGather.full_index)


class TestSelect:
    def test_select_batch(self, stream_tensor_bfl_fixture):
        """Test `torch.select` on a StreamTensor when applied to batch, length, and feature dimensions."""
        s = stream_tensor_bfl_fixture.select(dim=0, index=1)
        assert isinstance(s, StreamTensor)
        assert torch.equal(s.tensor(), stream_tensor_bfl_fixture.tensor().select(dim=0, index=1))
        assert s.meta.ids == ["middle"]
        assert torch.equal(s.meta.sos, torch.tensor([False]))
        assert torch.equal(s.meta.eos, torch.tensor([False]))
        assert torch.equal(s.meta.lengths, torch.tensor([3]))
        assert s.names == ("F", LENGTH)

    def test_select_feature(self, stream_tensor_bfl_fixture):
        s = stream_tensor_bfl_fixture.select(dim=1, index=1)
        assert isinstance(s, StreamTensor)
        assert torch.equal(s.tensor(), stream_tensor_bfl_fixture.tensor().select(dim=1, index=1))
        assert s.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s.meta.lengths, torch.tensor([3, 3, 2]))
        assert s.names == (BATCH, LENGTH)

    def test_select_length(self, stream_tensor_bfl_fixture):
        s = stream_tensor_bfl_fixture.select(dim=2, index=1)
        assert isinstance(s, StreamTensor)
        assert torch.equal(s.tensor(), stream_tensor_bfl_fixture.tensor().select(dim=2, index=1))
        assert s.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s.meta.sos, torch.tensor([False, False, False]))
        assert torch.equal(s.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s.meta.lengths, torch.tensor([1, 1, 1]))
        assert s.names == (BATCH, "F")


class TestTake:
    def test_take_batch(self, stream_tensor_bfl_fixture):
        # Linear indices that return the entire first batch example but drops other examples for shape (3, 2, 3)
        indices = torch.tensor([0, 1, 2, 3, 4, 5])
        s1 = stream_tensor_bfl_fixture.take(indices)
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor().take(indices))
        assert s1.meta.ids == ["first"]
        assert torch.equal(s1.meta.sos, torch.tensor([True]))
        assert torch.equal(s1.meta.eos, torch.tensor([False]))
        assert torch.equal(s1.meta.lengths, torch.tensor([3]))
        assert s1.names == (join_dim_names("F", LENGTH),)

        # Linear indices that return the entire first batch example but drops the first timestep and other examples
        indices = torch.tensor([1, 2, 4, 5])
        s2 = stream_tensor_bfl_fixture.take(indices)
        assert isinstance(s2, StreamTensor)
        assert torch.equal(s2.tensor(), stream_tensor_bfl_fixture.tensor().take(indices))
        assert s2.meta.ids == ["first"]
        assert torch.equal(s2.meta.sos, torch.tensor([False]))
        assert torch.equal(s2.meta.eos, torch.tensor([False]))
        assert torch.equal(s2.meta.lengths, torch.tensor([2]))
        assert s2.names == (join_dim_names("F", LENGTH),)

    def test_take_feature(self, stream_tensor_bfl_fixture):
        # Linear indices that return the entire first feature but drops second feature for shape (3, 2, 3)
        indices = torch.tensor([0, 1, 2, 6, 7, 8, 12, 13, 14])
        s1 = stream_tensor_bfl_fixture.take(indices)
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor().take(indices))
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([3, 3, 2]))
        assert s1.names == (join_dim_names(BATCH, "F", LENGTH),)

    def test_take_length(self, stream_tensor_bfl_fixture):
        # Linear indices that return the first length step for each example but drops other steps for shape (3, 2, 3)
        indices = torch.tensor([0, 3, 6, 9, 12, 15])
        s1 = stream_tensor_bfl_fixture.take(indices)
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor().take(indices))
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, False]))
        assert torch.equal(s1.meta.lengths, torch.tensor([1, 1, 1]))
        assert s1.names == (join_dim_names(BATCH, "F"),)

        # Linear indices that return the last length step for each example but drops other steps for shape (3, 2, 3)
        indices = torch.tensor([2, 5, 8, 11, 14, 17])
        s1 = stream_tensor_bfl_fixture.take(indices)
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor().take(indices))
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([False, False, False]))
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, False]))
        assert torch.equal(s1.meta.lengths, torch.tensor([1, 1, 0]))
        assert s1.names == (join_dim_names(BATCH, "F"),)

    def test_take_every_other(self, stream_tensor_bfl_fixture):
        # Linear indices that return every other element of the tensor for shape (3, 2, 3)
        indices = torch.tensor(range(0, stream_tensor_bfl_fixture.numel(), 2))
        s1 = stream_tensor_bfl_fixture.take(indices)
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor().take(indices))
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([3, 3, 2]))
        assert s1.names == (join_dim_names(BATCH, "F", LENGTH),)


class TestIndexSelect:
    def test_index_select_batch(self, stream_tensor_bfl_fixture):
        """Index select on the batch dim keeping only some examples but all other elements."""
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.index_select,
            dim=0,
            index=torch.tensor([0, 2]),
            ids=["first", "last"],
            sos=torch.tensor([True, False]),
            eos=torch.tensor([False, True]),
            lengths=torch.tensor([3, 2]),
            names=stream_tensor_bfl_fixture.names,
        )

    def test_index_select_feature(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture):
        """Index select on the feature dim keeping only some features but all other elements."""
        stream_meta_kwargs_fixture["names"] = (BATCH, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.index_select,
            dim=1,
            index=torch.tensor([0]),
            **stream_meta_kwargs_fixture,
        )

    def test_index_select_length(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture):
        """Index select on the length dim keeping only some lengths but all other elements."""
        stream_meta_kwargs_fixture["lengths"] = torch.tensor([2, 2, 1])
        stream_meta_kwargs_fixture["sos"] = torch.tensor([True, False, False])
        stream_meta_kwargs_fixture["eos"] = torch.tensor([False, False, True])
        stream_meta_kwargs_fixture["names"] = (BATCH, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.index_select,
            dim=2,
            index=torch.tensor([0, 2]),
            **stream_meta_kwargs_fixture,
        )

        with pytest.raises(IndexError):
            assert_stream_tensor_and_meta_correct(
                stream_tensor_bfl_fixture,
                torch.index_select,
                dim=2,
                index=torch.tensor([2, 0]),
                **stream_meta_kwargs_fixture,
            )


class TestMaskedSelect:
    def test_masked_select_batch(self, stream_tensor_bfl_fixture):
        """Masked select on the batch dim keeping only some examples but all other elements."""
        assert_fallback_to_torch_tensor(
            stream_tensor_bfl_fixture,
            torch.masked_select,
            mask=torch.tensor([True, False, True]).unsqueeze(1).unsqueeze(2),
        )

    def test_masked_select_feature(self, stream_tensor_bfl_fixture):
        """Masked select on the feature dim keeping only some features but all other elements."""
        assert_fallback_to_torch_tensor(
            stream_tensor_bfl_fixture,
            torch.masked_select,
            mask=torch.tensor([True, False]).unsqueeze(0).unsqueeze(2),
        )

    def test_masked_select_length(self, stream_tensor_bfl_fixture):
        """Masked select on the length dim keeping only some lengths but all other elements."""
        assert_fallback_to_torch_tensor(
            stream_tensor_bfl_fixture,
            torch.masked_select,
            mask=torch.tensor([False, False, True]).unsqueeze(0).unsqueeze(1),
        )


## Indexing (__getitem__)


class TestGetitem:
    def test_feature_indexing_integer(self, stream_tensor_bfl_fixture):
        """Indexing with an integer on the feature dim should remove the feature dim but not change the meta."""
        s1 = stream_tensor_bfl_fixture[:, 0, :]
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[:, 0, :])
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([3, 3, 2]))
        assert s1.names == (BATCH, LENGTH)

    def test_batch_indexing_integer(self, stream_tensor_bfl_fixture):
        """Indexing with an integer on the batch dim should remove the batch dim and change the meta to have only that
        one example's metadata."""
        s1 = stream_tensor_bfl_fixture[0]
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[0])
        assert s1.meta.ids == ["first"]  # changed to only the first example
        assert torch.equal(s1.meta.sos, torch.tensor([True]))
        assert torch.equal(s1.meta.eos, torch.tensor([False]))
        assert torch.equal(s1.meta.lengths, torch.tensor([3]))
        assert s1.names == ("F", LENGTH)

    def test_batch_indexing_slice(self, stream_tensor_bfl_fixture):
        """Indexing with a slice on the batch dim should remove relevant examples from the meta."""
        s1 = stream_tensor_bfl_fixture[1:]
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[1:])
        assert s1.meta.ids == ["middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([False, False]))
        assert torch.equal(s1.meta.eos, torch.tensor([False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([3, 2]))
        assert s1.names == (BATCH, "F", LENGTH)

    def test_batch_indexing_tuple(self, stream_tensor_bfl_fixture):
        s1 = stream_tensor_bfl_fixture[(0,)]
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[(0,)])
        assert s1.meta.ids == ["first"]
        assert torch.equal(s1.meta.sos, torch.tensor([True]))
        assert torch.equal(s1.meta.eos, torch.tensor([False]))
        assert torch.equal(s1.meta.lengths, torch.tensor([3]))
        assert s1.names == ("F", LENGTH)

        s2 = stream_tensor_bfl_fixture[(0, 1)]
        assert isinstance(s2, StreamTensor)
        assert torch.equal(s2.tensor(), stream_tensor_bfl_fixture.tensor()[(0, 1)])
        assert s2.meta.ids == ["first"]
        assert torch.equal(s2.meta.sos, torch.tensor([True]))
        assert torch.equal(s2.meta.eos, torch.tensor([False]))
        assert torch.equal(s2.meta.lengths, torch.tensor([3]))
        assert s2.names == (LENGTH,)

        s3 = stream_tensor_bfl_fixture[(0, 1, 2)]
        assert isinstance(s3, StreamTensor)
        assert torch.equal(s3.tensor(), stream_tensor_bfl_fixture.tensor()[(0, 1, 2)])
        assert s3.meta.ids == ["first"]
        assert torch.equal(s3.meta.sos, torch.tensor([False]))
        assert torch.equal(s3.meta.eos, torch.tensor([False]))
        assert torch.equal(s3.meta.lengths, torch.tensor([1]))
        assert s3.names == tuple()

        s4 = stream_tensor_bfl_fixture[(0, 1, 0)]
        assert isinstance(s4, StreamTensor)
        assert torch.equal(s4.tensor(), stream_tensor_bfl_fixture.tensor()[(0, 1, 0)])
        assert s4.meta.ids == ["first"]
        assert torch.equal(s4.meta.sos, torch.tensor([True]))
        assert torch.equal(s4.meta.eos, torch.tensor([False]))
        assert torch.equal(s4.meta.lengths, torch.tensor([1]))
        assert s4.names == tuple()

    def test_batch_indexing_list(self, stream_tensor_bfl_fixture):
        """Indexing with a list or tuple on the batch dim should remove relevant examples from the meta."""
        s1 = stream_tensor_bfl_fixture[[1]]
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[[1]])
        assert s1.meta.ids == ["middle"]
        assert torch.equal(s1.meta.sos, torch.tensor([False]))
        assert torch.equal(s1.meta.eos, torch.tensor([False]))
        assert torch.equal(s1.meta.lengths, torch.tensor([3]))
        assert s1.names == (BATCH, "F", LENGTH)

        s2 = stream_tensor_bfl_fixture[[0, 2]]
        assert isinstance(s2, StreamTensor)
        assert torch.equal(s2.tensor(), stream_tensor_bfl_fixture.tensor()[[0, 2]])
        assert s2.meta.ids == ["first", "last"]
        assert torch.equal(s2.meta.sos, torch.tensor([True, False]))
        assert torch.equal(s2.meta.eos, torch.tensor([False, True]))
        assert torch.equal(s2.meta.lengths, torch.tensor([3, 2]))
        assert s2.names == (BATCH, "F", LENGTH)

    def test_batch_indexing_booltensor(self, stream_tensor_bfl_fixture):
        """Indexing with a bool tensor on the batch dim should remove relevant examples from the meta."""
        s1 = stream_tensor_bfl_fixture[torch.tensor([False, True, True])]
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[torch.tensor([False, True, True])])
        assert s1.meta.ids == ["middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([False, False]))
        assert torch.equal(s1.meta.eos, torch.tensor([False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([3, 2]))
        assert s1.names == (BATCH, "F", LENGTH)

    def test_batch_indexing_inttensor(self, stream_tensor_bfl_fixture):
        """Indexing with an int tensor on the batch dim should remove relevant examples from the meta."""
        s1 = stream_tensor_bfl_fixture[torch.tensor([1, 2])]
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[torch.tensor([1, 2])])
        assert s1.meta.ids == ["middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([False, False]))
        assert torch.equal(s1.meta.eos, torch.tensor([False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([3, 2]))
        assert s1.names == (BATCH, "F", LENGTH)

    def test_length_indexing_integer(self, stream_tensor_bfl_fixture):
        """Indexing with an integer on the length dim should remove the length dim and change the meta to have lengths 1
        or 0 depending on padding, and sos only true when the index is 0, and eos only true when the index is the
        last non-padding index or beyond (TODO (JDH): Maybe we want eos False if in padding?).
        """
        s2 = stream_tensor_bfl_fixture[:, :, 0]  # first length index
        assert isinstance(s2, StreamTensor)
        assert torch.equal(s2.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, 0])
        assert s2.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s2.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s2.meta.eos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s2.meta.lengths, torch.tensor([1, 1, 1]))  # changed to 1
        assert s2.names == (BATCH, "F")

        s3 = stream_tensor_bfl_fixture[:, :, 1]  # middle length index
        assert isinstance(s3, StreamTensor)
        assert torch.equal(s3.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, 1])
        assert s3.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s3.meta.sos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s3.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s3.meta.lengths, torch.tensor([1, 1, 1]))  # changed to 1
        assert s3.names == (BATCH, "F")

        s4 = stream_tensor_bfl_fixture[:, :, -1]  # last length index
        assert isinstance(s4, StreamTensor)
        assert torch.equal(s4.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, -1])
        assert s4.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s4.meta.sos, torch.tensor([False, False, False]))  # change to False
        assert torch.equal(s4.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s4.meta.lengths, torch.tensor([1, 1, 0]))  # changed to 1 and 0
        assert s4.names == (BATCH, "F")

    def test_length_indexing_slice(self, stream_tensor_bfl_fixture):
        """Indexing with a slice on the length dim should correctly adjust the meta lengths and sos/eos.
        Specifically,
        - if the slice starts after the first index, no examples should be first.
        - if the slice ends before the last index, no examples should be last.
        - the lengths should be minus the number of non-padding tensor elements that are removed.
        """
        s2 = stream_tensor_bfl_fixture[:, :, 1:]  # remove first length index
        assert isinstance(s2, StreamTensor)
        assert torch.equal(s2.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, 1:])
        assert s2.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s2.meta.sos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s2.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s2.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all)
        assert s2.names == (BATCH, "F", LENGTH)

        s3 = stream_tensor_bfl_fixture[:, :, :-1]  # remove last length index
        assert isinstance(s3, StreamTensor)
        assert torch.equal(s3.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, :-1])
        assert s3.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s3.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s3.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s3.meta.lengths, torch.tensor([2, 2, 2]))  # minus 1 for all but "last" since it was padding
        assert s3.names == (BATCH, "F", LENGTH)

        s4 = stream_tensor_bfl_fixture[:, :, 1:-1]  # remove first and last length index
        assert isinstance(s4, StreamTensor)
        assert torch.equal(s4.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, 1:-1])
        assert s4.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s4.meta.sos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s4.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s4.meta.lengths, torch.tensor([1, 1, 1]))  # minus 2 for all but "last" since it was padding
        assert s4.names == (BATCH, "F", LENGTH)

        s5 = stream_tensor_bfl_fixture[:, :, :-2]  # remove two last length indices
        assert isinstance(s5, StreamTensor)
        assert torch.equal(s5.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, :-2])
        assert s5.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s5.meta.sos, torch.tensor([True, False, False]))  # changed to False
        assert torch.equal(s5.meta.eos, torch.tensor([False, False, False]))
        assert torch.equal(s5.meta.lengths, torch.tensor([1, 1, 1]))  # minus 2 for all but "last" since it was padding
        assert s5.names == (BATCH, "F", LENGTH)

        s6 = stream_tensor_bfl_fixture[:, :, ::2]  # remove every other length index from start to end
        assert isinstance(s6, StreamTensor)
        assert torch.equal(s6.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, ::2])
        assert s6.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s6.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s6.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s6.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all
        assert s6.names == (BATCH, "F", LENGTH)

    @pytest.mark.parametrize(
        "indices",
        [
            [list((1, 2)), list((0, 2)), list((0, 1))],
            [tuple((1, 2)), tuple((0, 2)), tuple((0, 1))],
        ],
    )
    def test_length_indexing_tuple_list(self, stream_tensor_bfl_fixture, indices):
        """"""
        s1 = stream_tensor_bfl_fixture[:, :, indices[0]]  # remove first length index
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, indices[0]])
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all
        assert s1.names == (BATCH, "F", LENGTH)

        s2 = stream_tensor_bfl_fixture[:, :, indices[1]]  # remove middle length index
        assert isinstance(s2, StreamTensor)
        assert torch.equal(s2.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, indices[1]])
        assert s2.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s2.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s2.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s2.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all
        assert s2.names == (BATCH, "F", LENGTH)

        s3 = stream_tensor_bfl_fixture[:, :, indices[2]]  # remove last length index
        assert isinstance(s3, StreamTensor)
        assert torch.equal(s3.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, indices[2]])
        assert s3.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s3.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s3.meta.eos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s3.meta.lengths, torch.tensor([2, 2, 2]))  # minus 1 for all but "last" since was padding
        assert s3.names == (BATCH, "F", LENGTH)

    def test_length_indexing_list(self, stream_tensor_bfl_fixture):
        """"""
        s1 = stream_tensor_bfl_fixture[:, :, [1, 2]]  # remove first length index
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, [1, 2]])
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all
        assert s1.names == (BATCH, "F", LENGTH)

        s2 = stream_tensor_bfl_fixture[:, :, [0, 2]]  # remove middle length index
        assert isinstance(s2, StreamTensor)
        assert torch.equal(s2.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, [0, 2]])
        assert s2.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s2.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s2.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s2.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all
        assert s2.names == (BATCH, "F", LENGTH)

        s3 = stream_tensor_bfl_fixture[:, :, [0, 1]]  # remove last length index
        assert isinstance(s3, StreamTensor)
        assert torch.equal(s3.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, [0, 1]])
        assert s3.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s3.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s3.meta.eos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s3.meta.lengths, torch.tensor([2, 2, 2]))  # minus 1 for all but "last" since it was padding
        assert s3.names == (BATCH, "F", LENGTH)

    def test_length_indexing_1d_inttensor(self, stream_tensor_bfl_fixture):
        """"""
        s1 = stream_tensor_bfl_fixture[:, :, torch.tensor([1, 2])]  # remove first length index
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, torch.tensor([1, 2])])
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all
        assert s1.names == (BATCH, "F", LENGTH)

        s2 = stream_tensor_bfl_fixture[:, :, torch.tensor([0, 2])]  # remove middle length index
        assert isinstance(s2, StreamTensor)
        assert torch.equal(s2.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, torch.tensor([0, 2])])
        assert s2.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s2.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s2.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s2.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all
        assert s2.names == (BATCH, "F", LENGTH)

        s3 = stream_tensor_bfl_fixture[:, :, torch.tensor([0, 1])]  # remove last length index
        assert isinstance(s3, StreamTensor)
        assert torch.equal(s3.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, torch.tensor([0, 1])])
        assert s3.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s3.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s3.meta.eos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s3.meta.lengths, torch.tensor([2, 2, 2]))  # minus 1 for all but "last" since it was padding
        assert s3.names == (BATCH, "F", LENGTH)

    def test_length_indexing_1d_booltensor(self, stream_tensor_bfl_fixture):
        """Test that we can index a StreamTensor with a bool tensor (mask)."""
        s1 = stream_tensor_bfl_fixture[:, :, torch.tensor([False, True, True])]  # remove first length index
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, torch.tensor([False, True, True])])
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all
        assert s1.names == (BATCH, "F", LENGTH)

        s2 = stream_tensor_bfl_fixture[:, :, torch.tensor([True, True, False])]  # remove last length index
        assert isinstance(s2, StreamTensor)
        assert torch.equal(s2.tensor(), stream_tensor_bfl_fixture.tensor()[:, :, torch.tensor([True, True, False])])
        assert s2.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s2.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s2.meta.eos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s2.meta.lengths, torch.tensor([2, 2, 2]))  # minus 1 for all but last since this was padding.
        assert s2.names == (BATCH, "F", LENGTH)

    def test_batch_and_feature_indexing_2d_booltensor(self, stream_tensor_bfl_fixture):
        """Test that we can index a StreamTensor with a 2d bool tensor along the batch and feature dimensions."""
        indices = torch.tensor([[True, True], [False, True], [True, False]])  # Keep only some features
        s1 = stream_tensor_bfl_fixture[indices]
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[indices])
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([3, 3, 2]))
        assert s1.shape == (4, 3)
        assert s1.names == (join_dim_names(BATCH, "F"), LENGTH)

    def test_length_and_feature_indexing_2d_booltensor(self, stream_tensor_bfl_fixture):
        """Test that we can index a StreamTensor with a 2d bool tensor along the length and feature dimensions."""
        # keep all lenghts of one feature and some of other
        indices = torch.tensor([[True, True, True], [True, False, False]])
        s1 = stream_tensor_bfl_fixture[:, indices]
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[:, indices])
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([3, 3, 2]))
        assert s1.names == (BATCH, join_dim_names("F", LENGTH))

        indices = torch.tensor([[True, False, False], [True, False, False]])  # remove last two time steps of features
        s2 = stream_tensor_bfl_fixture[:, indices]
        assert isinstance(s2, StreamTensor)
        assert torch.equal(s2.tensor(), stream_tensor_bfl_fixture.tensor()[:, indices])
        assert s2.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s2.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s2.meta.eos, torch.tensor([False, False, False]))
        assert torch.equal(s2.meta.lengths, torch.tensor([1, 1, 1]))  # changed to 1
        assert s2.names == (BATCH, join_dim_names("F", LENGTH))

    def test_batch_and_length_indexing_2d_booltensor(self, stream_tensor_lbf_fixture, stream_tensor_blf_fixture):
        """Test that we can index a StreamTensor with a 2d bool tensor along the length and feature dimensions."""
        # Remove last two steps of middle example
        indices = torch.tensor([[True, True, True], [True, False, False], [True, True, True]])  # (L, B) = (3, 3)
        assert_fallback_to_torch_tensor(stream_tensor_lbf_fixture, torch.Tensor.__getitem__, indices)

        # Remove middle example altogether
        indices = torch.tensor([[True, True, True], [False, False, False], [True, True, True]])  # (L, B) = (3, 3)
        assert_fallback_to_torch_tensor(stream_tensor_lbf_fixture, torch.Tensor.__getitem__, indices)

        # Remove middle example and first time step of first example and last two timesteps of last example
        indices = torch.tensor([[False, True, True], [False, False, False], [True, False, False]])  # (L, B) = (3, 3)
        assert_fallback_to_torch_tensor(stream_tensor_lbf_fixture, torch.Tensor.__getitem__, indices)

        # Transposed version of s3 (stream_tensor_blf instead of stream_tensor_lbf)
        # Remove middle timestep for all examples, include first timestep for middle and last, last timestep for first
        assert_fallback_to_torch_tensor(stream_tensor_blf_fixture, torch.Tensor.__getitem__, indices)

    def test_batch_and_length_indexing_3d_booltensor(self, stream_tensor_bfl_fixture):
        """Test indexing a StreamTensor with a 3D BoolTensor along the batch, feature and length dimensions."""
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

        assert_fallback_to_torch_tensor(stream_tensor_bfl_fixture, torch.Tensor.__getitem__, indices)

    def test_feature_indexing_2d_inttensor(self, stream_tensor_bfl_fixture):
        indices = torch.tensor([[0, 1], [1, 1]])
        s1 = stream_tensor_bfl_fixture[:, indices]
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[:, indices])
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([3, 3, 2]))
        assert s1.names == (BATCH, None, "F", LENGTH)
        assert s1.shape == (3, 2, 2, 3)

    def test_batch_indexing_2d_inttensor(self, stream_tensor_bfl_fixture):
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
        with pytest.raises(IndexError):
            stream_tensor_bfl_fixture[indices]

    def test_length_indexing_2d_inttensor(self, stream_tensor_bfl_fixture):
        indices = torch.tensor([[0, 1], [1, 1]])
        with pytest.raises(IndexError):
            stream_tensor_bfl_fixture[..., indices]

    def test_length_indexing_integer_multidimensional(self, stream_tensor_bfl_fixture):
        """Indexing with an integer on the length dim should remove the length dim and change the meta to have lengths 1
        or 0 depending on padding, and sos only true when the index is 0, and eos only true when the index is the last
        non-padding index or beyond (TODO (JDH): Maybe we want eos False if in padding?). This case also tests that we
        can simultaneously index the feature dimension without affecting the length indexing.
        """
        s1 = stream_tensor_bfl_fixture[:, 0, 0]  # first length index and first feature index
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[:, 0, 0])
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s1.meta.lengths, torch.tensor([1, 1, 1]))  # changed to 1
        assert s1.names == (BATCH,)

    def test_length_indexing_slice_multidimensional(self, stream_tensor_bfl_fixture):
        """Indexing with a slice on the length dim should correctly adjust the meta lengths and sos/eos.
        This case also tests that we can simultaneously index the feature dimension without affecting the length
        indexing.
        """
        s1 = stream_tensor_bfl_fixture[:, 0, 1:]  # remove first length index and first feature index
        assert isinstance(s1, StreamTensor)
        assert torch.equal(s1.tensor(), stream_tensor_bfl_fixture.tensor()[:, 0, 1:])
        assert s1.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s1.meta.sos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s1.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s1.meta.lengths, torch.tensor([2, 2, 1]))  # minus 1 for all
        assert s1.names == (BATCH, LENGTH)

    def test_batch_and_length_indexing_slice(self, stream_tensor_bfl_fixture):
        """Indexing with a slice on the batch dim and a slice on the length dim should have the combined effect of
        both."""
        # Remove last batch index and first length index
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.Tensor.__getitem__,
            (slice(None, -1), slice(None), slice(1, None)),  # [:-1, :, 1:]
            ids=["first", "middle"],
            sos=torch.tensor([False, False]),
            eos=torch.tensor([False, False]),
            lengths=torch.tensor([2, 2]),
            names=(BATCH, "F", LENGTH),
        )

    def test_indexing_ellipsis(self, stream_tensor_bfl_fixture):
        """Test that we can index a StreamTensor with an ellipsis."""
        s1_1 = stream_tensor_bfl_fixture[0, ...]  # keep only first batch example
        assert isinstance(s1_1, StreamTensor)
        assert torch.equal(s1_1.tensor(), stream_tensor_bfl_fixture.tensor()[0, ...])
        assert s1_1.meta.ids == ["first"]
        assert torch.equal(s1_1.meta.sos, torch.tensor([True]))
        assert torch.equal(s1_1.meta.eos, torch.tensor([False]))
        assert torch.equal(s1_1.meta.lengths, torch.tensor([3]))
        assert s1_1.names == ("F", LENGTH)

        s1_2 = stream_tensor_bfl_fixture[0, ..., :]  # keep only first batch example
        assert isinstance(s1_2, StreamTensor)
        assert torch.equal(s1_2.tensor(), stream_tensor_bfl_fixture.tensor()[0, ..., :])
        assert s1_2.meta.ids == ["first"]
        assert torch.equal(s1_2.meta.sos, torch.tensor([True]))
        assert torch.equal(s1_2.meta.eos, torch.tensor([False]))
        assert torch.equal(s1_2.meta.lengths, torch.tensor([3]))
        assert s1_1.names == ("F", LENGTH)

        s1_3 = stream_tensor_bfl_fixture[0, :, ...]  # keep only first batch example
        assert isinstance(s1_3, StreamTensor)
        assert torch.equal(s1_1.tensor(), stream_tensor_bfl_fixture.tensor()[0, :, ...])
        assert s1_3.meta.ids == ["first"]
        assert torch.equal(s1_3.meta.sos, torch.tensor([True]))
        assert torch.equal(s1_3.meta.eos, torch.tensor([False]))
        assert torch.equal(s1_3.meta.lengths, torch.tensor([3]))
        assert s1_1.names == ("F", LENGTH)

        s1_4 = stream_tensor_bfl_fixture[0, ..., ...]  # keep only first batch example
        assert isinstance(s1_4, StreamTensor)
        assert torch.equal(s1_1.tensor(), stream_tensor_bfl_fixture.tensor()[0, ..., ...])
        assert s1_4.meta.ids == ["first"]
        assert torch.equal(s1_4.meta.sos, torch.tensor([True]))
        assert torch.equal(s1_4.meta.eos, torch.tensor([False]))
        assert torch.equal(s1_4.meta.lengths, torch.tensor([3]))
        assert s1_1.names == ("F", LENGTH)

        s2 = stream_tensor_bfl_fixture[..., 0]  # keep only first length index
        assert isinstance(s2, StreamTensor)
        assert torch.equal(s2.tensor(), stream_tensor_bfl_fixture.tensor()[..., 0])
        assert s2.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s2.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s2.meta.eos, torch.tensor([False, False, False]))  # changed to False
        assert torch.equal(s2.meta.lengths, torch.tensor([1, 1, 1]))  # set to 1
        assert s2.names == (BATCH, "F")

        s3 = stream_tensor_bfl_fixture[..., 0, ...]  # keep only first feature dim
        assert isinstance(s3, StreamTensor)
        assert torch.equal(s3.tensor(), stream_tensor_bfl_fixture.tensor()[..., 0, ...])
        assert s3.meta.ids == ["first", "middle", "last"]
        assert torch.equal(s3.meta.sos, torch.tensor([True, False, False]))
        assert torch.equal(s3.meta.eos, torch.tensor([False, False, True]))
        assert torch.equal(s3.meta.lengths, torch.tensor([3, 3, 2]))
        assert s3.names == (BATCH, LENGTH)

    def test_indexing_none(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture):
        # Insert single new dimension before left most dimension.
        stream_meta_kwargs_fixture["names"] = (None, BATCH, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.Tensor.__getitem__,
            None,
            **stream_meta_kwargs_fixture,
        )

        # Insert two new dimensions before left most dimension.
        stream_meta_kwargs_fixture["names"] = (None, None, BATCH, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.Tensor.__getitem__,
            (None, None),
            **stream_meta_kwargs_fixture,
        )

        # Insert single new dimension at dim 1.
        stream_meta_kwargs_fixture["names"] = (BATCH, None, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.Tensor.__getitem__,
            (slice(None), None),
            **stream_meta_kwargs_fixture,
        )

        # Insert single new dimension at dim 2.
        stream_meta_kwargs_fixture["names"] = (BATCH, "F", None, LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.Tensor.__getitem__,
            (slice(None), slice(None), None),
            **stream_meta_kwargs_fixture,
        )

        # Insert single new dimension at dim 3 (right most dimension) using.
        stream_meta_kwargs_fixture["names"] = (BATCH, "F", LENGTH, None)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.Tensor.__getitem__,
            (slice(None), slice(None), slice(None), None),
            **stream_meta_kwargs_fixture,
        )

        # Insert single new dimension at dim 3 (right most dimension) using Ellipsis.
        stream_meta_kwargs_fixture["names"] = (BATCH, "F", LENGTH, None)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.Tensor.__getitem__,
            (..., None),
            **stream_meta_kwargs_fixture,
        )


class TestScatterAdd:
    batch_index = torch.tensor([[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]])
    feature_index = torch.tensor([[[0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]]])
    src = stream_tensor_bfl() * 2

    def test_scatter_add_batch(self, stream_tensor_bfl_fixture):
        assert_fallback_to_torch_tensor(
            stream_tensor_bfl_fixture, torch.scatter_add, dim=0, index=self.batch_index, src=self.src
        )

    def test_scatter_add_feature(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture):
        stream_meta_kwargs_fixture["names"] = (BATCH, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.scatter_add,
            dim=1,
            index=self.feature_index,
            src=self.src,
            **stream_meta_kwargs_fixture,
        )

    def test_scatter_add_length(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture):
        assert_fallback_to_torch_tensor(
            stream_tensor_bfl_fixture,
            torch.scatter_add,
            dim=2,
            index=self.feature_index,
            src=self.src,
        )

    def test_scatter_add_feature_inplace(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture):
        stream_meta_kwargs_fixture["names"] = (BATCH, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.Tensor.scatter_add_,
            dim=1,
            index=self.feature_index,
            src=self.src,
            **stream_meta_kwargs_fixture,
        )


@pytest.mark.parametrize("reduce", ["sum", "mean", "max", "min"])
class TestScatterReduce:
    batch_index = torch.tensor([[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]])
    feature_index = torch.tensor([[[0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]]])
    src = stream_tensor_bfl() * 2

    def test_scatter_reduce_batch(self, stream_tensor_bfl_fixture, reduce):
        assert_fallback_to_torch_tensor(
            stream_tensor_bfl_fixture,
            torch.scatter_reduce,
            dim=0,
            index=self.batch_index,
            src=self.src,
            reduce=reduce,
        )

    def test_scatter_reduce_feature(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture, reduce):
        stream_meta_kwargs_fixture["names"] = (BATCH, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.scatter_reduce,
            dim=1,
            index=self.feature_index,
            src=self.src,
            reduce=reduce,
            **stream_meta_kwargs_fixture,
        )

    def test_scatter_reduce_length(self, stream_tensor_bfl_fixture, reduce):
        assert_fallback_to_torch_tensor(
            stream_tensor_bfl_fixture,
            torch.scatter_reduce,
            dim=2,
            index=self.feature_index,
            src=self.src,
            reduce=reduce,
        )

    def test_scatter_reduce_feature_inplace(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture, reduce):
        stream_meta_kwargs_fixture["names"] = (BATCH, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.Tensor.scatter_reduce_,
            dim=1,
            index=self.feature_index,
            src=self.src,
            reduce=reduce,
            **stream_meta_kwargs_fixture,
        )


class TestFlatten:
    def test_flatten_batch(self, stream_tensor_bfld_fixture):
        assert_fallback_to_torch_tensor(stream_tensor_bfld_fixture, torch.flatten, start_dim=0, end_dim=1)

    def test_flatten_feature(self, stream_tensor_bfld_fixture, stream_meta_kwargs_fixture):
        stream_meta_kwargs_fixture["names"] = (BATCH, "F_D", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfld_fixture,
            torch.flatten,
            start_dim=1,
            end_dim=2,
            **stream_meta_kwargs_fixture,
        )

    def test_flatten_length(self, stream_tensor_bfld_fixture):
        assert_fallback_to_torch_tensor(stream_tensor_bfld_fixture, torch.flatten, start_dim=2, end_dim=3)


class TestUnsqueeze:
    def test_unsqueeze_single(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture):
        stream_meta_kwargs_fixture["names"] = (None, BATCH, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.unsqueeze,
            dim=0,
            **stream_meta_kwargs_fixture,
        )

    def test_unsqueeze_two(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture):
        stream_meta_kwargs_fixture["names"] = (None, None, BATCH, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture.unsqueeze(0),
            torch.unsqueeze,
            dim=0,
            **stream_meta_kwargs_fixture,
        )

    def test_unsqueeze_three(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture):
        stream_meta_kwargs_fixture["names"] = (None, None, BATCH, None, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture.unsqueeze(0).unsqueeze(2),
            torch.unsqueeze,
            dim=0,
            **stream_meta_kwargs_fixture,
        )


class TestSqueeze:
    def test_squeeze_non_singleton_dim(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture):
        stream_meta_kwargs_fixture["names"] = (BATCH, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            stream_tensor_bfl_fixture,
            torch.squeeze,
            dim=0,
            **stream_meta_kwargs_fixture,
        )

    def test_squeeze_singleton_dim(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture):
        s = as_stream_tensor(
            stream_tensor_bfl_fixture.tensor().sum(dim=1, keepdim=True),
            names=stream_tensor_bfl_fixture.names,
            meta=stream_tensor_bfl_fixture.meta,
        )
        stream_meta_kwargs_fixture["names"] = (BATCH, LENGTH)
        assert_stream_tensor_and_meta_correct(
            s,
            torch.squeeze,
            dim=1,
            **stream_meta_kwargs_fixture,
        )

    def test_squeeze_all_singleton_dims(self, stream_tensor_bfl_fixture, stream_meta_kwargs_fixture):
        names = (None, None) + stream_tensor_bfl_fixture.names
        s = as_stream_tensor(
            stream_tensor_bfl_fixture.unsqueeze(0).unsqueeze(0), names=names, meta=stream_tensor_bfl_fixture.meta
        )
        stream_meta_kwargs_fixture["names"] = (BATCH, "F", LENGTH)
        assert_stream_tensor_and_meta_correct(
            s,
            torch.squeeze,
            **stream_meta_kwargs_fixture,
        )
