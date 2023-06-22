import itertools
import math
import time

from typing import Generator, List, Callable

import pandas as pd
import pytest
import torch
import torchaudio

from dreamstream.data import AudioStreamDataset, MultiStreamDataLoader


NUM_TEST_FILES = 200
MAX_LENGTH = 1200
MIN_LENGTH = 20
SAMPLE_RATE = 40


class Compose:
    def __init__(self, *transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class Wait:
    def __call__(self, x):
        time.sleep(0.01)
        return x


@pytest.fixture(scope="session")
def filenames():
    return ["file_{:03d}.wav".format(i) for i in range(NUM_TEST_FILES)]


@pytest.fixture(scope="session")
def lengths():
    return [abs(hash(i)) % (MAX_LENGTH - MIN_LENGTH) + MIN_LENGTH for i in range(NUM_TEST_FILES)]


@pytest.fixture(scope="session")
def test_data_directory(tmp_path_factory, filenames, lengths):
    d = tmp_path_factory.mktemp("data")
    for filename, length in zip(filenames, lengths):
        fn = d / filename
        torchaudio.save(fn, torch.arange(length).unsqueeze(0).to(torch.float), SAMPLE_RATE)

    return d


@pytest.fixture(scope="session")
def filepaths(test_data_directory, filenames):
    return [test_data_directory / filename for filename in filenames]


@pytest.fixture(scope="session")
def test_source_df(filepaths, lengths):
    return pd.DataFrame({"filename": filepaths, "length": lengths})


def compute_num_samples(lengths, chunk_seconds):
    """Helper method to compute the number of samples in an AudioStreamDataset."""
    return sum([math.ceil(length / SAMPLE_RATE / chunk_seconds) for length in lengths])


class TestAudioStreamDataset:
    def test_instantiate(self, test_source_df):
        dataset = AudioStreamDataset(
            file_list=test_source_df.filename,
            file_lengths=test_source_df.length,
            chunk_seconds=1.0,
        )
        assert isinstance(dataset, AudioStreamDataset)
        assert isinstance(iter(dataset), Generator)

    @pytest.mark.parametrize("chunk_seconds", [0.1, 0.5, 1.0, 2.0])
    def test_iterate(self, test_source_df, chunk_seconds):
        dataset = AudioStreamDataset(
            file_list=test_source_df.filename,
            file_lengths=test_source_df.length,
            chunk_seconds=chunk_seconds,
        )
        examples = [example for example in dataset]
        num_samples = compute_num_samples(test_source_df.length, chunk_seconds)
        assert len(examples) == num_samples

    def test_automatic_get_metadata(self, test_source_df):
        dataset = AudioStreamDataset(
            file_list=test_source_df.filename,
            chunk_seconds=1.0,
        )
        assert dataset.file_metadata is not None
        assert len(dataset.file_metadata) == len(test_source_df)
        assert dataset.file_lengths is not None
        assert len(dataset.file_lengths) == len(test_source_df)

    @pytest.mark.parametrize(
        "use_file_lengths, num_workers, batch_size, shuffle, drop_last",
        itertools.product(
            *[
                [True, False],  # use_file_lengths
                [-1, 0, 1, 4, 10],  # num_workers
                [1, 10, 100],  # batch_size
                [True, False],  # shuffle
                [True, False],  # drop_last
            ]
        ),
    )
    def test_split(self, test_source_df, use_file_lengths, num_workers, batch_size, shuffle, drop_last):
        if use_file_lengths:
            dataset = AudioStreamDataset(test_source_df.filename, file_lengths=test_source_df.length, chunk_seconds=1.0)
        else:
            dataset = AudioStreamDataset(
                test_source_df.filename,
                chunk_seconds=1.0,
            )

        if (num_workers <= 0) or (batch_size % num_workers != 0):
            with pytest.raises(ValueError):
                dataset.split(num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
            return None

        datasets = dataset.split(num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        assert len(datasets) == num_workers
        assert all([isinstance(dataset, AudioStreamDataset) for dataset in datasets])
        assert all(isinstance(iter(dataset), Generator) for dataset in datasets)


class TestMultiStreamDataLoader:
    @pytest.mark.parametrize(
        "shuffle, drop_last, overlapping_batches",
        [
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ],
    )
    def test_instantiate(self, test_source_df, shuffle, drop_last, overlapping_batches):
        dataset = AudioStreamDataset(
            file_list=test_source_df.filename,
            file_lengths=test_source_df.length,
            chunk_seconds=1.0,
        )

        kwargs = {
            "batch_size": 10,
            "num_workers": 5,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "overlapping_batches": overlapping_batches,
        }

        if drop_last and not overlapping_batches:
            with pytest.raises(ValueError):
                MultiStreamDataLoader(dataset, **kwargs)
            return None

        dataloader = MultiStreamDataLoader(dataset, **kwargs)
        assert isinstance(dataloader, MultiStreamDataLoader)
        assert isinstance(iter(dataloader), Generator)

    @pytest.mark.parametrize(
        "num_workers, shuffle, overlapping_batches", itertools.product(*[[0, 1, 5], [True, False], [True, False]])
    )
    def test_iterate_drop_last0(self, test_source_df, num_workers, shuffle, overlapping_batches):
        dataset = AudioStreamDataset(
            file_list=test_source_df.filename,
            file_lengths=test_source_df.length,
            chunk_seconds=1.0,
        )
        loader = MultiStreamDataLoader(
            dataset,
            batch_size=10,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=False,
            overlapping_batches=overlapping_batches,
            collate_fn=dataset.custom_collate,
        )

        batches = []
        files_seen = set()
        files_started = set()
        files_ended = set()
        samples_seen = []
        for batch in loader:
            batches.append(batch)

            files_seen.update(batch.meta.ids)
            files_started.update(id for id, sos in zip(batch.meta.ids, batch.meta.sos) if sos == 1)
            files_ended.update(id for id, eos in zip(batch.meta.ids, batch.meta.eos) if eos == 1)

            sample_ids = [f"{batch.meta.ids[i]} {batch.meta.chunk_indices[i]:d}" for i in range(len(batch.meta.ids))]
            samples_seen.extend(sample_ids)

            sos = [int(sos) for sos in batch.meta.sos]
            if not overlapping_batches:
                assert all([sos[i] == sos[0] for i in range(len(sos))]), "All files must start simultaneously."

            # names = [id.split("/")[-1].split(".")[0] for id in batch.meta.ids]
            # chunk_idx = [chunk_indices for chunk_indices in batch.meta.chunk_indices]
            # eos = [int(eos) for eos in batch.meta.eos]
            # names_with_index = [f"{filename} {idx:2d}" for filename, idx in zip(names, chunk_idx)]
            # print(names_with_index, sos, eos)

        num_samples = compute_num_samples(test_source_df.length, chunk_seconds=1.0)
        assert files_started == files_seen, "Some files never started."
        assert files_ended == files_seen, "Some files never ended."
        assert len(test_source_df.filename) == len(files_seen), "Unexpected number of files seen."
        assert set(test_source_df.filename.apply(str)) == files_seen, "Not all files were seen."
        assert num_samples == len(samples_seen), "Unexpected number of samples seen."

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_iterate_drop_last1(self, test_source_df, shuffle):
        dataset = AudioStreamDataset(
            file_list=test_source_df.filename,
            file_lengths=test_source_df.length,
            chunk_seconds=1.0,
        )
        loader = MultiStreamDataLoader(
            dataset,
            batch_size=10,
            num_workers=0,
            shuffle=shuffle,
            drop_last=True,
            overlapping_batches=True,
            collate_fn=dataset.custom_collate,
        )

        batches = []
        files_seen = set()
        files_started = set()
        files_ended = set()
        samples_seen = []
        for batch in loader:
            batches.append(batch)

            files_seen.update(batch.meta.ids)
            files_started.update(id for id, sos in zip(batch.meta.ids, batch.meta.sos) if sos == 1)
            files_ended.update(id for id, eos in zip(batch.meta.ids, batch.meta.eos) if eos == 1)

            sample_ids = [f"{batch.meta.ids[i]} {batch.meta.chunk_indices[i]:d}" for i in range(len(batch.meta.ids))]
            samples_seen.extend(sample_ids)

        assert files_started == files_seen, "Some files never started."
        assert files_ended == files_seen, "Some files never ended."
        assert len(test_source_df.filename) >= len(files_seen), "Unexpected number of files seen."
        assert len(files_seen) > 0, "No files were seen."
