

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
