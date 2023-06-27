<p align="center">
  <img width="80%" src="torchstream-logo.png">
</p>
<br />


## Description

DreamStream is a Python package for making PyTorch models work efficiently in online settings such as speech recognition.

<br /><br />

<h3 align="center">
  :construction: WORK IN PROGRESS :construction:
</h3>


## Patching an `nn.Module`

DreamStream introduces a new method called `patch_module` which augments an `nn.Module` with the ability to process `StreamTensor`s. This ability is added to the module in the form of a new processing mode called `online` that is orthogonal to the existing `train` and `eval` modes.

```python
import dreamstream as ds

model = MyModel()
ds.patch_module(model)
```


## The `StreamTensor` object

DreamStream introduces one primary data structure called a `StreamTensor` which is a subclass of `torch.Tensor`.


## Behaviour matrix

| Module mode      | `Tensor`                                                                             | `StreamTensor`                                                                          |
|-----------------|:------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|
| <span style="color:LightCoral"><b>Offline</b></span> - Train | <span style="color:green;font-size:18px"><b>&#10003;</b></span> Standard behavoiur | <span style="color:red;font-size:18px"><b>&#10007;</b></span>   Not supported          |
| <span style="color:LightCoral"><b>Offline</b></span> - Eval  | <span style="color:green;font-size:18px"><b>&#10003;</b></span> Standard behavoiur | <span style="color:red;font-size:18px"><b>&#10007;</b></span>   Not supported          |
| <span style="color:DarkSeaGreen"><b>Online</b></span> - Train  | <span style="color:red;font-size:18px"><b>&#10007;</b></span> Not supported      | <span style="color:green;font-size:18px"><b>&#10003;</b></span> DreamStream behaviour |
| <span style="color:DarkSeaGreen"><b>Online</b></span> - Eval   | <span style="color:red;font-size:18px"><b>&#10007;</b></span> Not supported      | <span style="color:green;font-size:18px"><b>&#10003;</b></span> DreamStream behaviour |


**DreamStream behaviour**

If a module has been patched with `patch_module` and then set to `online`, DreamStream augments the standard PyTorch behaviour to support online processing. 
This means, that any call to `.forward()` now expects that all `torch.Tensor` inputs are replaced by `StreamTensor` inputs.

The patched module, and its child modules, will then keep buffers of the internal state of the module for each `StreamTensor` input keyed by the `StreamTensor`'s `StreamMetaData.ids`.

We support <span style="color:DarkSeaGreen"><b>online</b></span> processing using `StreamTensor`s both for the `train` and `eval` modes. 

> <span style="color:DarkSeaGreen"><b>Online</b></span> - **Eval**
>
> If the module is <span style="color:DarkSeaGreen"><b>online</b></span> and in `eval` mode, any call to forward ...

> <span style="color:DarkSeaGreen"><b>Online</b></span> - **Train**
>
> If the module is <span style="color:DarkSeaGreen"><b>online</b></span> in `train` mode, any call to forward ...
> 
> We support `train` mode in the following variations:
>
> 1. **Chunk-wise backpropagation with aligned targets**: If `patch_module` was called with `train_variant="chunk-wise-aligned"`, a full forward-backward pass is performed on each chunk but with the forward pass conditioned on the detached (`.detach()`) state of the previous chunk that was forwarded on this `id`. This requires that each chunk of a larger file has targets, i.e. a single file level target is not supported. This mode is constant memory complexity in terms of the number of chunks and therefore enables training a large files. However, it does not backpropagate gradients through the entire file but only within each chunk.
> 2. **Full backpropagation with aligned targets**: If `patch_module` was called with `train_variant="full-file-aligned"`, a forward pass is performed on each chunk with the forward pass conditioned on the state (*not detached*) of the previous chunk that was forward on this `id`. For each chunk we compute the loss which requires chunk-aligned targets as for 1, and after the an entire file has been processed, we perform a backward pass on the total loss. This mode only provides memory savings compared to naively forward-backward passing the entire file at once if the model has superlinear memory complexity in terms of the sequence length. This mode is therefore not recommended for models with linear memory complexity such as RNNs and CNNs.
> 3. **Full backpropagation with unaligned targets**: If `patch_module` was called with `train_variant="full-file-unaligned"`, behaviour is like 2, but we accept a single target for a file (*do not require chunk-aligned targets*). Instead, from the chunk-wise forward calls, we accumulate the outputs needed to compute the total loss on the file level. Once a file has ended, we then compute the total loss and perform a backward pass. As for 2., this mode is only recommended for models with linear memory complexity such as RNNs and CNNs.
> 4. **Standard training**: If the `StreamTensor` passed to `.forward()` represents the entire file (i.e. has all `sos` and all `eos` True), the <span style="color:DarkSeaGreen"><b>online</b></span> `train` mode behaves like the standard PyTorch <span style="color:LightCoral"><b>offline</b></span> `train` mode, as could be expected. This will be the behaviour regardless of the `train_variant` defined in `patch_module`.


## TorchScript and ONNX

DreamStream supports TorchScript and ONNX export of patched modules. To compile a model that has DreamStream behaviour, the scripting, tracing and exporting must be done on a module in <span style="color:DarkSeaGreen"><b>online</b></span> mode. If in <span style="color:LightCoral"><b>offline</b></span> mode, the model would be exported as a standard PyTorch model. 

Exporting the patched model using the `dynamic_axes=False` argument to `torch.onnx.export` will export the model such that it works for a fixed chunk size. This is the most common case since models are usually served for streaming using a constant chunk size. Alternatively, if `dynamic_axes=True`, the model will be exported such that it works for any chunk size. This is useful if the model is to be used for streaming with variable chunk sizes but comes at the cost of a performance penalty.


## Interface

```python
ds.patch_module
ds.StreamModule  # automatically patches itself after __init__

ds.stream_tensor
ds.StreamTensor

ds.meta
ds.StreamMetadata

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
- How do we deal with operations that are invalid on one or more StreamTensors?
  - Examples:
    - Adding two StreamTensors with different `ids`.
    - Combining batch or length dimensions with one or more other dimensions into a single dimension using e.g. `torch.reshape`, `torch.flatten` or masked indexing.
  - Options:
    - Fail outright.
    - Fallback to a regular `torch.Tensor`.
    - Fallback to a different tensor subclass that is identical in behaviour to `torch.Tensor` but carries the frozen `StreamMetadata` along.
- How do we deal with 
  - Special tokens concatenated to the input? E.g. "translate" and "language" tokens in Whisper?
  - Learnable tokens concatenated to the input sequence before an MHSA layer?

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
