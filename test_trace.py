from typing import Dict, Tuple, Union
from random import random

import torch


class SkipConv1d(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv1d(*args, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + x
        


class Model(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.conv1 = SkipConv1d(4, 4, 5, 1, padding=2)
        self.conv2 = SkipConv1d(4, 4, 5, 1, padding=2)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        
        if random() < 0.5:
            x = -x
        
        return x

x = torch.randn(1, 4, 10)
model = Model()
graph = torch.fx.Tracer().trace(model)

for x in graph.nodes:
    print(x)