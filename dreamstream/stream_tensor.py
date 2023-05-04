import torch


class StreamTensor(torch.Tensor):
    
    @staticmethod 
    def __new__(cls, x, stream_state, *args, **kwargs): 
        return super().__new__(cls, x, *args, **kwargs) 
    
    def __init__(self, x, stream_state, *args, **kwargs):
        self._stream_state = stream_state
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        stream_states = [x._stream_state for x in args if isinstance(x, StreamTensor)]
        # TODO: If more than one, assert that stream_states are identical, or raise error
        out = super().__torch_function__(func, types, args, kwargs)
        if isinstance(out, torch.Tensor):
            return StreamTensor(out, stream_state=stream_states[0])
        return out
    
    @property
    def stream_state(self):
        return self._stream_state
        
    @stream_state.setter
    def stream_state(self, s):
        self._stream_state = s
