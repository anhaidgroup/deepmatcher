import torch


class AlignmentNetwork(type, layers=2, hidden_size=None, layer_module=dm.nn.Highway):
    pass


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, *args):
        return self.lambd(*args)
