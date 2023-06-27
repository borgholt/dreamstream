from dreamstream.tensor import LazyInit, LazyProxy


class TestObject():
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class LazyTestObject(LazyInit):
    def __init__(self, *args, **kwargs):
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
