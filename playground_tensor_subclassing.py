from typing import Any
import torch


class OwnedTensor(torch.Tensor):
    owner = "Minstry of Silly Walks"
    
    # def __new__(cls, *args, **kwargs):
    #     return super().__new__(cls, *args, **kwargs)

    # def __init__(self, *args, **kwargs):
    #     super().__init__()
    #     self.owner = "Minstry of Silly Walks"

    # def set_owner(self, owner):
    #     self.owner = owner

    def __getattribute__(self, __name: str) -> Any:
        print(f"__getattribute__({__name=})")
        return super().__getattribute__(__name)

    def __repr__(self):
        return f"{super().__repr__()} owned by {self.owner}"


if __name__ == "__main__":
    t = OwnedTensor([[1, 2, 3], [4, 5, 6]])

    o1 = t.clone()  # meta is not preserved for StreamTensor but it is here?
    o2 = t.transpose(0, 1)
    o3 = torch.rand_like(t)

    print(f"{t=}")

    print(f"{o1=}")
    print(f"{o2=}")
    print(f"{o3=}")

    import IPython
    IPython.embed(using=False)
