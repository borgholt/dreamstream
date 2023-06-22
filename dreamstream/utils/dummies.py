# from typing import List, Callable

# import torch

# from dreamstream import StreamTensor

# class TestMetadata:
#     """Metadata associated with a batch of streamed input tensors."""

#     def __init__(
#         self,
#         property="this is a test",
#     ):
#         self.property = property

# class TestTensor(StreamTensor):
    
#     # @staticmethod
#     # def __new__(cls, data, meta: TestMetadata, *args, **kwargs) -> "TestTensor":
#     #     """Return a new StreamTensor object."""
#     #     return super().__new__(cls, data, *args, **kwargs)

#     # def __init__(self, data, meta: TestMetadata, *args, names: List[str] = None, **kwargs):
#     #     """Initialize a StreamTensor object (self is StreamTensor, data is e.g. torch.Tensor)."""
#     #     super(TestTensor).__init__()
#     #     self.meta = meta
        
#     def clone(self, *args, **kwargs):
#         """Clone a StreamTensor object."""
#         return TestTensor(super().clone(*args, **kwargs), self.meta)
    
#     @classmethod
#     def __torch_function__(cls, func: Callable, types: List[torch.Tensor], args=(), kwargs=None):
#         return torch.Tensor.__torch_function__(cls, func, types, args, kwargs)

# class TestObjects():
    
#     a = torch.rand(2, 3, 4)
#     b = a.rename("B", "F", "L")
#     c = TestTensor(a, TestMetadata("this is another test"))
#     d = TestTensor(b, TestMetadata("this is another test"))
    
# test = TestObjects()