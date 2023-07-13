

class TorchStreamError(Exception):
    """Base class for exceptions in this module."""
    pass


class TorchStreamIndexError(TorchStreamError, IndexError):
    """Indexing error for StreamTensors."""
    pass
