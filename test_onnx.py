
from typing import Any, Callable, List, Self, Tuple, Union

import torch


class TestTensor(torch.Tensor):
    
    meta = None
    
    @classmethod
    def __torch_function__(cls, func: Callable, types: List[torch.Tensor], args=(), kwargs=None):
        
        if kwargs is None:
            kwargs = {}
        
        if func is torch.sigmoid:
            return custom_sigmoid(*args, **kwargs)
        
        return super().__torch_function__(func, types, args, kwargs)
    
    def tensor(self) -> torch.Tensor:
        return torch.Tensor(self)
    
def custom_sigmoid(x: TestTensor) -> TestTensor:
    meta = x.meta
    x = torch.sigmoid(x.tensor())
    meta = meta // 2
    x = TestTensor(x)
    x.meta = meta
    return x

class Model(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: TestTensor, meta: torch.Tensor) -> Tuple[TestTensor, torch.Tensor]:
        x.meta = meta
        x = torch.sigmoid(x)
        meta = x.meta
        return x, meta


model = Model()
x: TestTensor = TestTensor(torch.randn(2, 3))
meta = torch.arange(5)
y0 = model(x, meta)

scripted_model = torch.jit.script(model)
y1 = scripted_model(x)