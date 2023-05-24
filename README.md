<p align="center">
  <img width="65%" src="dreamstream-logo.png">
</p>
<br />


## Description

DreamStream is a Python package for making PyTorch models work efficiently in online settings such as speech recognition.

<br /><br />

<h3 align="center">
  :construction: WORK IN PROGRESS :construction:
</h3>


## Interface

```python
ds.patch_module
ds.StreamModule  # automatically patches itself after __init__

ds.stream_tensor
ds.StreamTensor

ds.stream_state
ds.StreamState

ds.ChunkModule  # maybe functional similar to ds.patch_module



class MyVerySpecialModel(StreamModule):
  def __init__(self, asd, asd,asd):
    super().__init__()
    self.linear = nn.Linear(10, 10)


mm = MyVerySpecialModel()


class StreamModule(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def when_init_is_done(self):
    ds.patch_module(self)
```



## What problem does this solve?

PyTorch models are typically trained and evaluated on batches of data. However, in online settings, such as speech recognition, data is often streamed in one sample at a time. This means that the model must be able to process a single sample at a time, and that the model must be able to process the sample as soon as it is received. This is not possible with the standard PyTorch API, which requires the entire example to be collected before the model can be evaluated.


## Value proposition

- Real-time processing: Return output as soon as the model has processed the input, reducing latency.
- Scalability: Reduce memory requirements for modules with superlinear memory complexity (e.g. transformers) by processing input in smaller samples.


## Use-cases

1. Transcribe a large number of files stored on disk (for example). The files are already fully available, but the files may be too long to process in full length due to memory.
2. Transcribe a live audio stream. The audio is not available in full length, but must be processed as it is received. Such a stream could be a live audio stream from a microphone, or a stream of audio chunks from a network connection.

## What concerns are there about this approach?

- Lack of compatibility with existing PyTorch inference frameworks (ONNX, TorchScript, etc.)


## Implementation problems

- Padding of first sample among a batch of samples, but not others (`torch.roll`)
- Handling samples in a batch that reach zero length at some layer during a forward pass.
- Efficient file loading when processing files in bulk (workers get copies of the dataset, but must not read the same files).
  - (`worker_init_fn`, `torch.utils.data.get_worker_info()`)
  - --> Sequential sampler, dataset, dataloader with single worker.
- Passing tensor with sequence lengths, first, last, and id among layers without breaking:
  - Batch object with data, sl, first, last, id attributes: Elementwise operations implemented outside Modules will break.
  - Subclass of Tensor with sl, first, last, id attributes: Elementwise operations implemented outside Modules will work but usually return new Tensor objects without these attributes.
    - This can work by implementing a custom `__torch_function__` method that returns a new Tensor object with the attributes.
    - We probably still need to deal with:
      - operations that reorder the batch dimension (e.g. `torch.sort` or indexing): Apply the same reordering to the attributes. This happens in `collate_fn` and before `torch.nn.utils.rnn.pack_padded_sequence`.
      - operations such as `cat`, `stack`, `vstack`, and `hstack` when used along the batch dim: Concatenate also the attributes along the batch dim. This happens in `collate_fn` but could also happen in models.
      - operations that reduce the batch dimension (e.g. `torch.sum`): Reduce also the attributes along the batch dim. This happens in losses.
      - operations that create new dimensions: Adjust `batch_dim` and `stream_dim` accordlingly. This could happen anywhere.
  - Seperate arguments to forward method: We can change all patched forward methods to take additional arguments, but, by default they won't be given (and we have no control).

## Can we use DreamStream for training?

- Case 1: Forward pass each chunk and compute loss. Detach module stream buffers after each module forward. Backpropagate and update parameters. This is O(1) memory in terms of the number of chunks for convolutions and RNNs but O(N^2) for Transformers.
- Case 2: Forward pass each chunk and collect logits, but do not detach stream buffers. Concatenate logits and compute loss. Backpropagate. This will backpropagate through the entire stream. This is less memory efficient for convolutions and RNNs O(N), in terms of the number of chunks.


## Cool features

- Patched module can provide estimate of state size (can be used to estimate memory requirements for given input lengths).
- Length-based sampling: When processing files in bulk, first sort files by length to minimize padding.
- Maybe we can support the EmFormer architecture.


## Documentation

Documentation is written in Sphinx and is inspired by [PyTorch](https://github.com/pytorch/pytorch/tree/main/docs).


## Installation

```bash
conda deactivate
conda create -y -n dreamstream python==3.11
conda activate dreamstream
pip install --upgrade --editable . 
pip install -r requirements.txt
```



```python

class Input():
  def __init__(self, data):
    self.data = data
    self.first = first
    self.last = last


class Batch(Iterable):
  def __init__(self, inputs: List[Input]):
    super().__init__(data)
    self.inputs = inputs
    self.is_collated = False

  def append(self, input: Input):
    self.inputs.append(input)
    self.is_collated = False

  def extend(self, inputs: Union[List[Input], Batch]):
    self.inputs.extend(inputs)
    self.is_collated = False

  def collate(self):
    self.data = collate_fn([input.data for input in self.inputs])
    self.is_collated = True
    return self


def stream_forward(x: Union[Single, Batch]):
  if not x.is_collated:
    x.collate()

  if not self.is_streaming:
    return self.original_forward(x.data)

  # ... streamnig logic
  x = self.original_forward(x.data)
  # ... more streaming logic
  return x

```
