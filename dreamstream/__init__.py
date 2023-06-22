import dreamstream.overrides  # noqa: F401
import dreamstream.random  # noqa: F401

from dreamstream.tensor import StreamTensor, StreamMetadata, stream_tensor, as_stream_tensor  # noqa: F401
from dreamstream.patches import patch, patch_module, add_streaming_modes, patch_conv_1d  # noqa: F401
