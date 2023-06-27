import time

from dreamstream.tensor import LazyInit, LazyProxy
from dreamstream.utils.timing import timeit


class TestObject():
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        time.sleep(0.1)


class LazyTestObject(LazyInit):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs


class TestLazy():
    def test_lazy_proxy_laziness(self):
        lazy_proxy = LazyProxy(TestObject, 1, 2, 3, a=1, b=2, c=3)
        assert lazy_proxy.__dict__["_cls"] is TestObject
        assert lazy_proxy.__dict__["_args"] == (1, 2, 3)
        assert lazy_proxy.__dict__["_kwargs"] == {'a': 1, 'b': 2, 'c': 3}
        assert lazy_proxy.__dict__["_obj"] is None

    def test_lazy_proxy_initialization(self):
        lazy_proxy = LazyProxy(TestObject, 1, 2, 3, a=1, b=2, c=3)
        _ = lazy_proxy.args  # trigger initialization
        assert isinstance(lazy_proxy.__dict__["_obj"], TestObject)
        assert lazy_proxy.__dict__["_obj"].args == (1, 2, 3)
        assert lazy_proxy.__dict__["_obj"].kwargs == {'a': 1, 'b': 2, 'c': 3}
        
    def test_lazy_init(self):
        lazy_init = LazyTestObject(1, 2, 3, a=1, b=2, c=3)
        assert isinstance(lazy_init, LazyProxy)

    def test_lazy_init_time_saved(self):
        """Test that lazy initialization indeed saves time."""
        lazy_timing = timeit("LazyTestObject(1, 2, 3, a=1, b=2, c=3)", globals=globals())
        init_timing = timeit("TestObject(1, 2, 3, a=1, b=2, c=3)", globals=globals())
        assert lazy_timing.median < init_timing.median / 100
