

def online(self, mode=True):
    if not isinstance(mode, bool):
        raise ValueError("streaming mode is expected to be boolean")
    self.streaming = mode
    for module in self.children():
        module.online(mode)
    return self

def offline(self):
    return self.online(mode=False)


def patch(module):
    raise NotImplementedError("stream_patch is not implemented")