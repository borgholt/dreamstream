import random
import torch

from torchaudio.models import Wav2Letter

from dreamstream.patches.general import patch
from dreamstream.utils.flags import LENGTH
from dreamstream.nn.utils import pad_full_sequence, pad_stream_tensor
from dreamstream.data import OutputCollector


def random_chunks(full_length, min_size: int = 1000, max_size: int = 8000):
    chunks = []
    chunk_sum, remaining = 0, full_length
    while remaining > 0:
        chunks.append(min(random.randint(min_size, max_size), remaining))
        chunk_sum = sum(chunks)
        remaining = full_length - chunk_sum
    return chunks


def run():
    random.seed(42)
    torch.manual_seed(42)

    BATCH_SIZE = 32

    model = Wav2Letter(num_classes=40, input_type="waveform")
    model = patch(model)

    # Test with multiple streams of different lengths.
    sequences = [torch.rand(1, random.randint(16000 * 5, 16000 * 30)) for i in range(BATCH_SIZE)]

    ids = [str(hash(i)) for i in range(BATCH_SIZE)]
    targets = {_id: model(s.unsqueeze(0)) for _id, s in zip(ids, sequences)}

    data = pad_full_sequence(sequences, names=("F", LENGTH), ids=ids).align_to("B", "F", "L")
    data = data.unpad_sequence()
    data = {
        _id: s.split(random_chunks(s.size("L"), min_size=16000, max_size=32000), dim=1) for _id, s in zip(ids, data)
    }

    def remaining_chunks(data):
        return sum([len(x) for x in data.values()])

    batches = []
    while remaining_chunks(data) > 0:
        batch = [s.pop(0) for _id, s in data.items() if len(s) > 0 and random.random() < 0.75]
        if len(batch) > 0:
            batches.append(pad_stream_tensor(batch).align_to("B", "F", "L"))

    stream_output = OutputCollector()
    model.online()
    ys = []
    for x in batches:
        y = model(x)
        ys.append(y)
        stream_output.update(y)

    for _id, _y in targets.items():
        y = stream_output[_id].tensor()
        abs_diff = (_y - y).abs()
        print(
            _id,
            y.shape,
            torch.allclose(_y, y, atol=1e-6),
            abs_diff.max().item(),
            abs_diff.max(0).values[:10].max().item(),
        )


if __name__ == "__main__":
    run()
