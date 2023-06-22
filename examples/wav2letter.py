import os
import random
import pandas as pd
import pathlib
import requests
import torch
import uuid

from torchaudio.models import Wav2Letter

from dreamstream.data import AudioStreamDataset, MultiStreamDataLoader

from dreamstream.patches.general import patch
from dreamstream.utils.flags import LENGTH
from dreamstream.nn.utils import pad_full_sequence, pad_stream_tensor
from dreamstream.patches import patch_conv_1d
from dreamstream.data import OutputCollector


# _SAMPLE_DIR = pathlib.Path(os.path.join(os.path.dirname(__file__), "sample_data"))
# _SAMPLE_DIR.mkdir(exist_ok=True)


# SAMPLE_WAV_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.wav"
# SAMPLE_WAV_PATH = _SAMPLE_DIR / "steam.wav"

# SAMPLE_WAV_SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
# SAMPLE_WAV_SPEECH_PATH = _SAMPLE_DIR / "speech.wav"

# SAMPLE_RIR_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/distant-16k/room-response/rm1/impulse/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo.wav"
# SAMPLE_RIR_PATH = _SAMPLE_DIR / "rir.wav"

# SAMPLE_NOISE_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/distant-16k/distractors/rm1/babb/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav"
# SAMPLE_NOISE_PATH = _SAMPLE_DIR / "bg.wav"


# def _fetch_data():
#   uri = [
#     (SAMPLE_WAV_URL, SAMPLE_WAV_PATH),
#     (SAMPLE_WAV_SPEECH_URL, SAMPLE_WAV_SPEECH_PATH),
#     (SAMPLE_RIR_URL, SAMPLE_RIR_PATH),
#     (SAMPLE_NOISE_URL, SAMPLE_NOISE_PATH),
#   ]
#   for url, path in uri:
#       if not os.path.exists(path):
#             with open(path, 'wb') as file_:
#                 file_.write(requests.get(url).content)
    
    
def random_chunks(full_length, min_size: int = 1000, max_size: int = 8000):
    chunks = []
    chunk_sum, remaining = 0, full_length
    while remaining > 0:
        chunks.append(min(random.randint(min_size, max_size), remaining))
        chunk_sum = sum(chunks)
        remaining = full_length - chunk_sum
    return chunks


def run():
    # _fetch_data()
    
    # source_files = [SAMPLE_WAV_PATH, SAMPLE_WAV_SPEECH_PATH, SAMPLE_RIR_PATH, SAMPLE_NOISE_PATH]
    # dataset = AudioStreamDataset(source_files, chunk_seconds=0.3)
    # loader = MultiStreamDataLoader(dataset, batch_size=2, num_workers=0, overlapping_batches=False, collate_fn=dataset.custom_collate)
    

    # model = Wav2Letter(num_classes=40, input_type="waveform")
    # model = torch.nn.Conv1d(1, 128, 3, padding=1)
    model = torch.nn.Sequential(
        torch.nn.Conv1d(1, 128, 3, padding=1),
        torch.nn.Conv1d(128, 256, 5, padding=2),
    )
    model = patch(model)

    # TEST 1: Test with multiple streams of different lengths.
    sequences = [torch.rand(1, random.randint(16000, 48000)) for i in range(32)]

    ids = [str(uuid.uuid4()) for i in range(32)]
    targets = {_id: model(s) for _id, s in zip(ids, sequences)}
    # original_sequences = {_id: s for _id, s in zip(ids, sequences)}

    data = pad_full_sequence(sequences, names=("F", LENGTH), ids=ids).align_to("B", "F", "L")
    data = data.unpad_sequence()
    data = {_id: s.split(random_chunks(s.size("L"), min_size=16000, max_size=32000), dim=1) for _id, s in zip(ids, data)}


    def remaining_chunks(data):
        return sum([len(x) for x in data.values()])


    batches = []
    while remaining_chunks(data) > 0:
        batch = [s.pop(0) for _id, s in data.items() if len(s) > 0 and random.random() < 0.75]
        if len(batch) > 0:
            batches.append(pad_stream_tensor(batch).align_to("B", "F", "L"))

    stream_output = OutputCollector()
    model.online()
    for x in batches:
        print(x.shape)
        y = model(x)
        stream_output.update(y)

    for _id, _y in targets.items():
        y = stream_output[_id].tensor()
        abs_diff = (_y - y).abs()
        print(y.size(-1), torch.allclose(_y, y), abs_diff.max().item(), abs_diff.max(0).values[:10].max().item())


if __name__ == "__main__":
    run()
