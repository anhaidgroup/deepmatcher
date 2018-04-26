class Bunch:

    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def tally_parameters(model):
    return sum([p.nelement() for p in model.parameters() if p.requires_grad])
